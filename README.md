# Alpha Animals
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository contains research and algorithms for our team, Alpha Animals, in IMC Prosperity 2025. We placed [9th globally and 2nd in the USA](https://jmerle.github.io/imc-prosperity-3-leaderboard/), with an overall score of 1,190,077 seashells. This writeup was heavily inspired by the [linear utility writeup](https://github.com/ericcccsliu/imc-prosperity-2).

## The team ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/cartert27">
          <img src="https://avatars.githubusercontent.com/u/67401383?v=4?s=100" width="100px;" alt="Carter Tran"/>
          <br /><sub><b>Carter Tran</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/cartertran/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-cartert27" title="Research">🔬</a>
        <a href="https://github.com/cartert27/imc-prosperity-3/commits?author=cartert27" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/kenjigunawan">
          <img src="https://avatars.githubusercontent.com/u/174416052?v=4?s=100" width="100px;" alt="Kenji Gunawan"/>
          <br /><sub><b>Kenji Gunawan</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/kenjigunawan/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-kenjigunawan" title="Research">🔬</a>
        <a href="https://github.com/cartert27/imc-prosperity-3/commits?author=kenjigunawan" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/evanai23">
          <img src="https://avatars.githubusercontent.com/u/147462126?v=4?s=100" width="100px;" alt="Evan Ai"/>
          <br /><sub><b>Evan Ai</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/evanai/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-evanai23" title="Research">🔬</a>
        <a href="https://github.com/cartert27/imc-prosperity-3/commits?author=evanai23" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/1marcb">
          <img src="https://avatars.githubusercontent.com/u/175369812?v=4?s=100" width="100px;" alt="Marc Boundames"/>
          <br /><sub><b>Marc Boudames</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/marc-boudames1/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-1marcb" title="Research">🔬</a>
        <a href="https://github.com/cartert27/imc-prosperity-3/commits?author=1marcb" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/ssyquia">
          <img src="https://avatars.githubusercontent.com/u/54983967?v=4?s=100" width="100px;" alt="Sancho Syquia"/>
          <br /><sub><b>Sancho Syquia</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/ssyquia/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-ssyquia" title="Research">🔬</a>
        <a href="https://github.com/cartert27/imc-prosperity-3/commits?author=ssyquia" title="Code">💻</a>
      </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

Here is our island, which looks suspiciously like La Jolla, CA—where you’ll find both beautiful beaches and the UC San Diego campus. It’s possible our local expertise in seashell gathering gave us a competitive edge.

![9th Place Island](9th_place.png)

## The competition 🏆


IMC Prosperity 2025 was an algorithmic trading competition that lasted over 15 days, with over 13,500 teams participating globally. In the challenge, we were tasked with algorithmically trading various products, such as rainforest resin, kelp, squid ink, volcanic rock, and more, with the goal of maximizing seashells: the underlying currency of our island. We started trading rainforest resin, kelp, and squid ink in round 1, and with each subsequent round, more products were added. At the end of each round, our trading algorithm was evaluated against bot participants in the marketplace, whose behavior we could try and predict through historical data. The PNL from this independent evaluation would then be compared against all other teams. 

In addition to the main algorithmic trading focus, the competition also consisted of manual trading challenges in each round. The focus of these varied widely, and in the end, manual trading accounted for just a small fraction of our PNL. 

For documentation on the algorithmic trading environment, and more context about the competition, feel free to consult the [Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4). 

## Organization 📂

This repository contains some of our research notebooks, the data for each round, and our algorithm contained in `trader.py`. 

<details>
<summary><h2>Tools 🛠️</h2></summary>

We relied heavily on the open source [backtester](https://github.com/jmerle/imc-prosperity-3-backtester) and [visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer) built by [jmerle](https://github.com/jmerle). 

### Backtester 🔙

The open-source backtester saved us a lot of development time, but we had a few qualms. First, the logging was too verbose, so we were unable to visualize our later runs from AWS Lambda errors. In round 4, we were able to trade with a foreign island using **conversions**, which were unsupported by the backtester. Unable to locally test our strategy properly, we misunderstood the trading mechanics and ended up losing a lot of profit to conversion costs. 

### Dashboard 💨

The visualizer allowed us to walk through our algorithm one timestamp at a time, visualize the algorithms in terms of PnL, position sizing, order book, and more. It was very helpful for debugging and investigating the details of our algorithm's performance. 
</details>

<details>
<summary><h2>Round 1️⃣</h2></summary>

**Products:** Rainforest resin, kelp, squid ink

**Strategy:**
For Kelp, we implemented a market making strategy that identified large bids and asks from consistent market makers to reduce noise from small orders. Our strategy tracked their mid-price to determine a reliable fair value, placing limit orders around this price with configurable spreads. We also incorporated opportunistic taking of orders when they appeared mispriced relative to our fair value.

For Rainforest Resin, we used a simplified market making approach with a hardcoded fair value of 10000. This allowed us to avoid complexities of price discovery in a relatively stable product while still capturing spread from market making.

For Squid Ink, we implemented a short-term volatility spike mean-reversion strategy. We would detect price movements that exceeded 3 standard deviations from a 10-timestamp moving window and take positions in the opposite direction of these movements, betting on the price reverting to the mean. The strategy also included position management rules to limit exposure time and risk.

After Round 1, we were ranked 207 in the world. 

</details>

<details>
<summary><h2>Round 2️⃣</h2></summary>

**Products:** Picnic basket 1 and 2, jams, croissants, djembes

**Strategy:**
For Picnic Baskets, we implemented a statistical arbitrage strategy. We calculated the fair value of each basket using a linear model with coefficients for their component products (Croissants, Jams, and Djembes). When the market price of a basket diverged significantly from our calculated synthetic value, we attempted to trade to capture this difference.

However, due to a bug in our algorithm that caused it to try buying more than the position limit, our statistical arbitrage strategy wasn't functioning correctly. We decided to focus our efforts on other products rather than debugging this error, so we didn't actively trade the component products (Jams, Croissants, Djembes) during this round.

Despite these issues, our performance in Round 2 helped our team advance to rank 58.

</details>

<details>
<summary><h2>Round 3️⃣</h2></summary>

**Products:** Volcanic rock, volcanic rock vouchers

**Strategy:**
For Volcanic Rock and its vouchers, we implemented an options pricing strategy using the Black-Scholes model. We treated vouchers as call options on Volcanic Rock with various strike prices. The strategy calculates implied volatility from market prices, maintains a rolling volatility window, and prices vouchers based on this volatility estimate. We also look for arbitrage opportunities between vouchers with different strike prices, exploiting situations where the price spread between vouchers deviates from their strike price differences.

For Volcanic Rock itself, we implemented a strategy that uses the average implied volatility from all vouchers to determine if the underlying rock is fairly priced. When the rock price significantly deviates from our model's prediction, we take directional positions.

Due to an unexpected bug in our code, we ended up shorting volcanic rock at the max position limit for the entire duration of the trading day. Fortunately for us, this strategy ended up working, bringing our ranking to 2nd in the world! 

</details>

<details>
<summary><h2>Round 4️⃣</h2></summary>

**Products:** Magnificent macarons

**Strategy:**
For Magnificent Macarons, we implemented a cross-market arbitrage strategy that considers trading with a foreign island through the conversion mechanism. The strategy analyzes the bid/ask spreads in both local and foreign markets, accounting for transportation fees, import/export tariffs, and other conversion costs to identify profitable arbitrage opportunities.

Our approach dynamically adapts to market conditions based on the "sunlight index" observation:
- In normal sunlight regime: We execute two-way arbitrage, buying locally and selling abroad when local prices are lower than foreign prices (after accounting for fees), and vice versa.
- In low sunlight regime: We switch to an accumulation strategy, aggressively building up long positions while avoiding exports, capitalizing on the favorable conditions for holding inventory.

The strategy also includes market making elements, placing competitive bids and asks to provide liquidity while skewing order sizes based on our current position to maintain balance.

We couldn't get a working macarons strategy by the end of this round, and we decided to disable trading volcanic rock until we could get a proper strategy. Luckily enough for us, we remained 2nd in the world, while almost every other team in the top 30 dropped. 

</details>

<details>
<summary><h2>Round 5️⃣</h2></summary>

**New Change:** Counterparties are revealed

**Strategy:**
When counterparties were revealed, we suspected there might be an insider trader with an information advantage in the marketplace. To systematically identify this trader, we implemented a data-driven approach. We calculated the percentage of "good trades" (buying at low prices and selling at high prices relative to future price movements) for each trader over rolling windows. By filtering traders who consistently executed advantageous trades at a rate significantly above chance, we were able to narrow down potential insiders. We then visualized these traders' buy and sell orders relative to the mid-price of each asset and observed that a trader named "Olivia" was suspiciously precise in her timing—buying just before price increases and selling just before price drops across multiple products. This confirmed our hypothesis that she was indeed an insider with advance knowledge of price movements.

With counterparties revealed, we implemented a copy trading strategy specifically for Squid Ink and Croissants by tracking trades made by "Olivia". When Olivia bought one of these products, we interpreted this as a bullish signal and also bought; when she sold, we took this as a bearish signal and sold alongside her. This approach helped us establish market regimes (bullish or bearish) based on insider behavior.

We did not actively trade Jams or Djembes in this round, focusing instead on the copy trading strategy for Croissants and Squid Ink, where we could leverage Olivia's insider information.

For the Magnificent Macarons, we made a quick fix to our regime modeling strategy by ignoring the sunlight index completely. Instead, we focused solely on statistical arbitrage throughout the entire trading day, as we weren't confident whether our sunlight threshold was overfit to previous data or not.

For other products, we maintained our existing strategies:
- Market making for Kelp and Rainforest Resin
- Statistical arbitrage for Picnic Baskets
- Black-Scholes model for Volcanic Rock Vouchers

We also implemented a position management system that scales order sizes based on current inventory to avoid overexposure, and added features to close positions for inactive products to reduce risk.

Unfortunately our luck ran dry, and we were not able to make up the ground from the previous two rounds, and ended up being surpassed by a few teams. We ended at 9th in the world and 2nd in the USA, behind CMU Physics. Overall, we're very thankful to have had the opportunity to compete in this competition, and we are pleased with the results for this being many of our first times competing in a trading competition. 

</details>

<details>
<summary><h2>Other things we tried</h2></summary>

Throughout the competition, we experimented with several approaches that ultimately didn't make it into our final strategy:

- **Basket Mean-Reversion Modeling**: We attempted to model the spread between picnic baskets and their synthetic prices (based on component values) for mean-reversion trading. While theoretically sound, this approach faced implementation challenges and didn't yield consistent results.

- **Volatility Surface Fitting**: For volcanic rock vouchers, we tried fitting a volatility surface (smile) to the implied volatility versus moneyness from the call options. This would have given us a more sophisticated options pricing model, but it proved too complex to implement reliably given the competition's time constraints.

- **Delta Hedging**: We attempted to implement delta hedging with volcanic rock positions to create market-neutral strategies, but struggled with proper calibration and execution within the position limits.

- **Component-Basket Arbitrage**: While we developed code for direct arbitrage between picnic baskets and their component products (Croissants, Jams, Djembes), we never successfully traded the components due to implementation bugs and position limit challenges.

- **Price Pattern Analysis**: We spent considerable time modeling the price movements of virtually every product (squid ink, croissants, jams, djembes, volcanic rock, macarons, etc.) to find correlations, seasonality, or other patterns in the data. Despite extensive analysis, many of these efforts didn't yield actionable strategies.

- **Insider Signal Integration**: After identifying Olivia as an insider trader, we attempted to use her directional hints as a regime indicator to adjust our bids/asks dynamically, rather than just copying her trades. This proved more complicated than direct copy trading and didn't provide sufficiently reliable signals to justify the added complexity.

</details>

---

TLDR: This year's IMC Prosperity was very similar to the last two years, and we were able to successfully adapt and build-upon previous winning open-source strategies for this year's competition.