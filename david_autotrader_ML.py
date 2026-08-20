# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
#

"""Delta-neutral ETF/future market maker.

Fair value is an EWMA of the future's microprice. Quotes sit a vol-scaled
half-spread either side, skewed against inventory; ETF fills are hedged flat
in the future. A UCB1 bandit scales the half-spread per volatility regime,
rewarded on realised 1s markout.
"""

import asyncio
import itertools
import math

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side
from collections import deque
import time

# Exchange limits.
ACTIVE_ORDER_COUNT_LIMIT = 10
ACTIVE_VOLUME_LIMIT = 200
POSITION_LIMIT = 100

TICK_SIZE_IN_CENTS = 100
THEO_ALPHA = 0.35           # EWMA weight per future update

SKEW = 1                    # cents of quote shift per lot held
LOT_SIZE = 10

# Fast/slow pairs; their ratio is the regime signal.
VOL_FAST_A, VOL_SLOW_A = 0.10, 0.01
VLM_FAST_A, VLM_SLOW_A = 0.10, 0.01

BASE_HALF_SPREAD, MIN_HALF_SPREAD, MAX_HALF_SPREAD = 200, 100, 600

SPREAD_K = 1.3              # spread sensitivity to vol
BASE_LOT, MIN_LOT, MAX_LOT = 10, 2, 20
WARMUP = 30

# ---- bandit ----
ARMS = (0.8, 1.0, 1.4)      # half-spread multipliers
N_STATES = 3                # calm / normal / fast
UCB_C = 0.5                 # exploration
ARM_HOLD = 0.5              # seconds per episode
MARKOUT_H = 1.0             # reward horizon
REWARD_SCALE = float(BASE_HALF_SPREAD * BASE_LOT)


class Bandit:
    """UCB1 over spread multipliers, one table per market state."""

    def __init__(self, n_states, n_arms, c=UCB_C):
        self.c = c
        self.n = [[0] * n_arms for _ in range(n_states)]    # pull counts
        self.q = [[0.0] * n_arms for _ in range(n_states)]  # mean reward
        self.t = [0] * n_states

    def select(self, s):
        for a, n in enumerate(self.n[s]):
            if n == 0:
                return a                # try each arm once first
        lt = math.log(self.t[s])
        return max(range(len(self.n[s])),
                   key=lambda a: self.q[s][a] + self.c * math.sqrt(lt / self.n[s][a]))

    def update(self, s, a, r):
        self.n[s][a] += 1
        self.t[s] += 1
        self.q[s][a] += (r - self.q[s][a]) / self.n[s][a]   # incremental mean


class AutoTrader(BaseAutoTrader):
    """Quotes the ETF around a future-derived fair value and hedges the delta."""

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.orders = {}                # id -> {side, price, remaining, ep, fees}
        self.etf_position = self.future_position = 0
        self.msg_times = deque()        # rate limiter
        self.theo = None                # EWMA fair value
        self.future_micro = self.etf_micro = None
        self.pending_hedge = {}         # hedge_id -> signed lots in flight
        self.recent_trades_etf = deque()

        self.prev_micro = None
        self.vol_fast = self.vol_slow = None
        self.vlm_fast = self.vlm_slow = None
        self.n_obs = 0

        self.markout_q = deque()        # fills awaiting 1s scoring
        self.fees_total = 0

        # ---- bandit state ----
        self.bandit = Bandit(N_STATES, len(ARMS))
        self.episodes = {}              # ep_id -> [state, arm, reward, close_time]
        self.ep_ids = itertools.count(1)
        self.ep_id = None
        self.ep_end = 0.0
        self.arm_mult = 1.0

    @staticmethod
    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    @property
    def active_volume(self):
        """Lots resting in the book."""
        return sum(o['remaining'] for o in self.orders.values())

    @property
    def delta(self):
        """Net exposure across both legs; the hedger drives this to zero."""
        return self.etf_position + self.future_position

    @property
    def vol_ratio(self):
        """Volatility against its recent baseline; 1.0 is normal."""
        if self.n_obs < WARMUP or not self.vol_slow:
            return 1.0
        return self.clip(self.vol_fast / max(self.vol_slow, 1e-6), 0.3, 4.0)

    @property
    def vlm_ratio(self):
        """Traded volume against its recent baseline; 1.0 is normal."""
        if self.vlm_slow is None or self.vlm_slow < 1e-6:
            return 1.0
        return self.clip(self.vlm_fast / self.vlm_slow, 0.3, 3.0)

    def _half_spread(self):
        """Quote distance from theo: vol-scaled, then scaled by the bandit's arm."""
        hs = BASE_HALF_SPREAD * (1 + SPREAD_K * (self.vol_ratio - 1)) * self.arm_mult
        return self.clip(hs, MIN_HALF_SPREAD, MAX_HALF_SPREAD)

    def _lot_size(self, side):
        """Bigger when calm and busy, zero at the position limit."""
        vol_f = self.clip(1.0 / self.vol_ratio, 0.5, 1.5)
        liq_f = self.clip(self.vlm_ratio, 0.7, 1.3)
        room = (POSITION_LIMIT - self.etf_position) if side == Side.BUY \
                else (POSITION_LIMIT + self.etf_position)
        inv_f = self.clip(room / POSITION_LIMIT, 0.0, 1.3)
        size = BASE_LOT * vol_f * liq_f * inv_f
        return int(self.clip(round(size), 0, min(MAX_LOT, room)))

    def _can_send(self, now):
        """Sliding 1s window against the message rate limit."""
        while self.msg_times and now - self.msg_times[0] > 1.0:
            self.msg_times.popleft()
        return len(self.msg_times) < 45     # 45, not 50 - margin for in-flight messages

    def _note_send(self, now):
        self.msg_times.append(now)

    def _update_vol(self, micro):
        """Feed one absolute move into the vol EWMAs."""
        if self.prev_micro is not None:
            d = abs(micro - self.prev_micro)
            if self.vol_fast is None:
                self.vol_fast = self.vol_slow = d
            else:
                self.vol_fast += VOL_FAST_A * (d - self.vol_fast)
                self.vol_slow += VOL_SLOW_A * (d - self.vol_slow)
            self.n_obs += 1
        self.prev_micro = micro

    @staticmethod
    def _microprice(bid_px, bid_vol, ask_px, ask_vol):
        """Volume-weighted mid; leans toward the thinner side."""
        if bid_px == 0 or ask_px == 0:
            return None
        total = bid_vol + ask_vol
        if total == 0:
            return (bid_px + ask_px) / 2.0
        return (bid_px * ask_vol + ask_px * bid_vol) / total

    # ---- bandit ----

    def _state(self):
        """Bucket vol_ratio into calm / normal / fast."""
        vr = self.vol_ratio
        return 0 if vr < 0.9 else (1 if vr < 1.3 else 2)

    def _episode(self, now):
        """Live episode, opening a new one when the arm expires. Orders carry its id."""
        if self.ep_id is not None and now < self.ep_end:
            return self.ep_id
        s = self._state()
        a = self.bandit.select(s)
        self.arm_mult = ARMS[a]
        self.ep_id = next(self.ep_ids)
        self.ep_end = now + ARM_HOLD
        # Stays open past its end so late markouts still land in it.
        self.episodes[self.ep_id] = [s, a, 0.0, self.ep_end + MARKOUT_H + 0.5]
        return self.ep_id

    def _close_episodes(self, now):
        """Train on episodes whose fills can no longer change."""
        for eid in [e for e, d in self.episodes.items() if now > d[3]]:
            s, a, r, _ = self.episodes.pop(eid)
            self.bandit.update(s, a, r)

    # ---- order management ----

    def _try_insert(self, side, price, volume, now, ep=None):
        """Post an order if every limit allows it."""
        if volume <= 0 or price <= MINIMUM_BID or price >= MAXIMUM_ASK:
            return False
        if price % TICK_SIZE_IN_CENTS != 0 or len(self.orders) >= ACTIVE_ORDER_COUNT_LIMIT:
            return False
        if self.active_volume + volume > ACTIVE_VOLUME_LIMIT or not self._can_send(now):
            return False

        order_id = next(self.order_ids)
        self.send_insert_order(order_id, side, price, volume, Lifespan.GOOD_FOR_DAY)
        self.orders[order_id] = {'side': side, 'price': price, 'remaining': volume,
                                 'ep': ep, 'fees': 0}
        self._note_send(now)
        return True

    def _try_cancel(self, order_id, now):
        """Pull an order if it is live and the rate limit has room."""
        if order_id not in self.orders or not self._can_send(now):
            return False
        self.send_cancel_order(order_id)
        self._note_send(now)
        return True

    def _hedge(self, now):
        """Flatten delta in the future. Extreme limits make these market orders."""
        effective = self.delta + sum(self.pending_hedge.values())
        if effective == 0 or not self._can_send(now):
            return
        volume = abs(effective)
        if effective > 0:       # long overall -> sell futures
            side, price, signed = Side.SELL, MINIMUM_BID, -volume
        else:                   # short overall -> buy futures
            side, price, signed = Side.BUY, MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS, volume

        hedge_id = next(self.order_ids)
        self.send_hedge_order(hedge_id, side, price, volume)
        self.pending_hedge[hedge_id] = signed
        self._note_send(now)

    def _requote(self, now):
        """Reprice both sides, replacing only the orders that moved."""
        if self.theo is None:
            return

        ep = self._episode(now)
        hs = self._half_spread()
        centre = self.theo - SKEW * self.etf_position   # long -> quote lower
        bid = int((centre - hs) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
        ask = int(-((-(centre + hs)) // TICK_SIZE_IN_CENTS)) * TICK_SIZE_IN_CENTS

        # Never quote inside 50c of theo, however far skew pushed the centre.
        bid = min(bid, int((self.theo - 50) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS)
        ask = max(ask, int(-((-(self.theo + 50)) // TICK_SIZE_IN_CENTS)) * TICK_SIZE_IN_CENTS)

        want = {}
        if self.etf_position < POSITION_LIMIT:
            want[Side.BUY] = bid
        if self.etf_position > -POSITION_LIMIT:
            want[Side.SELL] = ask

        live = {}
        for oid, o in list(self.orders.items()):
            if want.get(o['side']) == o['price']:
                live[o['side']] = oid
            else:
                self._try_cancel(oid, now)

        for side, price in want.items():
            if side not in live:
                size = self._lot_size(side)
                self._try_insert(side, price, size, now, ep)

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Order rejected: drop our record so state stays real"""
        self.logger.warning("ERROR order %d: %s", client_order_id, error_message.decode())
        self.orders.pop(client_order_id, None)
        self.pending_hedge.pop(client_order_id, None)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Apply a hedge fill. Zero volume means it failed."""
        signed = self.pending_hedge.pop(client_order_id, 0)
        if volume == 0:
            self.logger.warning("hedge %d FAILED", client_order_id)
            return
        self.future_position += volume if signed > 0 else -volume
        self.logger.info("HEDGE px=%d vol=%d fut=%d delta=%d",
                         price, volume, self.future_position, self.delta)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Main loop. Future updates drive fair value, scoring, learning and requoting."""
        micro = self._microprice(bid_prices[0], bid_volumes[0], ask_prices[0], ask_volumes[0])
        if micro is None:
            return

        now = time.time()
        if instrument == Instrument.FUTURE:
            self._update_vol(micro)
            self.future_micro = micro
            self.theo = micro if self.theo is None else (THEO_ALPHA * micro + (1 - THEO_ALPHA) * self.theo)

            now = time.time()

            # Score aged fills: edge is spread captured, mo is adverse selection.
            while self.markout_q and self.markout_q[0][0] <= now:
                _, side, px, vol, theo0, ep = self.markout_q.popleft()
                sign = 1 if side == Side.BUY else -1
                edge = sign * (theo0 - px)
                mo   = sign * (self.theo - theo0)
                if ep in self.episodes:
                    self.episodes[ep][2] += (edge + mo) * vol / REWARD_SCALE
                self.logger.info("MARKOUT %s vol=%d edge=%.1f mo=%.1f net=%.1f fees=%d",
                                 "B" if side == Side.BUY else "S",
                                 vol, edge, mo, edge + mo, self.fees_total)

            self._close_episodes(now)
            self._requote(now)

        else:
            self.etf_micro = micro

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Update inventory, queue the fill for scoring, hedge if delta is large."""
        o = self.orders.get(client_order_id)
        if o is None:
            self.logger.warning("FILL for unknown order %d vol=%d — POSITION MAY BE WRONG", client_order_id, volume)
            return
        self.etf_position += volume if o['side'] == Side.BUY else -volume
        self.logger.info("FILL %s px=%d vol=%d theo=%.1f etf=%d delta=%d",
                         o['side'], price, volume, self.theo or 0.0,
                         self.etf_position, self.delta)
        now = time.time()
        # theo and ep travel with the fill so the reward can be attributed later.
        self.markout_q.append((now + MARKOUT_H, o['side'], price, volume, self.theo, o.get('ep')))
        if abs(self.delta) >= 10:
            self._hedge(now)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Track remaining size and accumulate fees (negative when we earn the rebate)."""
        o = self.orders.get(client_order_id)
        if o is not None:
            self.fees_total += fees - o.get('fees', 0)   # reported cumulatively
            o['fees'] = fees
        if remaining_volume == 0:
            self.orders.pop(client_order_id, None)
        elif o is not None:
            o['remaining'] = remaining_volume

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Traded ETF volume feeds the liquidity EWMAs."""
        if instrument != Instrument.ETF:
            return
        v = float(sum(ask_volumes) + sum(bid_volumes))
        if self.vlm_fast is None:
            self.vlm_fast = self.vlm_slow = v
        else:
            self.vlm_fast += VLM_FAST_A * (v - self.vlm_fast)
            self.vlm_slow += VLM_SLOW_A * (v - self.vlm_slow)