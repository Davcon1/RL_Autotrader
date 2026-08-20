Delta-neutral ETF/future market maker, built in Optiver's RTG environment.

Fair value is an EWMA of the future's microprice. Quotes sit a vol-scaled half-spread either side, skewed against inventory; ETF fills are hedged flat in the future. 

A UCB1 bandit scales the half-spread per volatility regime, rewarded on realised 1s markout.
