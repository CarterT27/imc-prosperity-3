from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
import json
import math
import statistics
import pandas as pd
import numpy as np
import jsonpickle
import typing


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # Maintain price history for calculating fair values
        self.kelp_prices = []
        self.resin_prices = []
        self.squid_ink_prices = []
        self.croissants_prices = []
        self.jams_prices = []
        self.djembes_prices = []
        self.kelp_vwap = []
        self.resin_vwap = []
        self.squid_ink_vwap = []
        self.croissants_vwap = []
        self.jams_vwap = []
        self.djembes_vwap = []
        
        # Trading toggles for each product
        self.active_products = {
            "KELP": True,
            "RAINFOREST_RESIN": True,
            "SQUID_INK": True,
            "CROISSANTS": True,
            "JAMS": True,
            "DJEMBES": True,
            "PICNIC_BASKET1": True,  # Now active
            "PICNIC_BASKET2": True   # Now active
        }
        
        # Position limits for each product - more conservative for volatile products
        self.position_limits = {
            "KELP": 50,  # Reduced due to high volatility
            "RAINFOREST_RESIN": 50,  # Stable product, can handle larger positions
            "SQUID_INK": 50,  # Use full position limit for reversion trades
            "CROISSANTS": 250,  # New product with high limit
            "JAMS": 350,  # New product with high limit
            "DJEMBES": 60,  # Changed from DJEMBE to DJEMBES
            "PICNIC_BASKET1": 60,  # Will handle separately
            "PICNIC_BASKET2": 100   # Will handle separately
        }
        
        # Parameters for trading strategies
        self.timespan = 20  # Increased history for better pattern detection
        self.make_width = {
            "KELP": 8.0,  # Wider spread due to volatility
            "RAINFOREST_RESIN": 3.0,  # Tighter spread due to stability
            "SQUID_INK": 5.0,  # Medium spread
            "CROISSANTS": 1.0,  # Using KELP-like strategy
            "JAMS": 2.0,  # Using KELP-like strategy
            "DJEMBES": 2.0  # Changed from DJEMBE to DJEMBES
        }
        self.take_width = {
            "KELP": 1.0,  # More aggressive for volatile market
            "RAINFOREST_RESIN": 0.3,  # Conservative taking strategy
            "SQUID_INK": 0.7,  # Balanced approach
            "CROISSANTS": 0.5,  # Using KELP-like strategy
            "JAMS": 0.5,  # Using KELP-like strategy
            "DJEMBES": 0.5  # Changed from DJEMBE to DJEMBES
        }
        
        # SQUID_INK specific parameters
        self.squid_ink_volatility_threshold = 3.0  # Increased threshold for taking positions
        self.squid_ink_momentum_period = 10
        self.squid_ink_mean_window = 30  # Window for calculating average price
        self.squid_ink_deviation_threshold = 0.05  # 5% deviation from mean triggers trade
        self.squid_ink_max_position_time = 5  # Maximum time to hold a position
        self.squid_ink_position_start_time = 0  # Track when we entered a position
        self.squid_ink_last_position = 0  # Track our last position for exit strategy

    def calculate_fair_value(
        self, order_depth: OrderDepth, method="mid_price", min_vol=0
    ) -> float:
        """Calculate fair value of a product based on order book."""
        if method == "mid_price":
            if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
                return None
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif method == "mid_price_with_vol_filter":
            filtered_asks = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= min_vol
            ]
            filtered_bids = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= min_vol
            ]

            if len(filtered_asks) == 0 or len(filtered_bids) == 0:
                return self.calculate_fair_value(order_depth, "mid_price")

            best_filtered_ask = min(filtered_asks)
            best_filtered_bid = max(filtered_bids)
            return (best_filtered_ask + best_filtered_bid) / 2

    def clear_position_order(
        self,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ) -> tuple[int, int]:
        """Add orders to clear position when close to fair value."""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = int(round(fair_value))
        fair_for_bid = int(math.floor(fair_value))
        fair_for_ask = int(math.ceil(fair_value))

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # If we're long, try to sell at fair price
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(
                    order_depth.buy_orders[fair_for_ask], position_after_take
                )
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        # If we're short, try to buy at fair price
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(
                    abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take)
                )
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def product_orders(
        self, product: str, order_depth: OrderDepth, position: int, state_timestamp: int = 0
    ) -> list[Order]:
        """Generate orders for a specific product."""
        orders = []
        position_limit = self.position_limits[product]
        
        # Use specialized strategy for SQUID_INK
        if product == "SQUID_INK":
            return self.squid_ink_strategy(order_depth, position, state_timestamp)

        buy_order_volume = 0
        sell_order_volume = 0

        # Skip if there are no orders in the market
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders

        # Calculate fair value based on current market
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Find market maker orders (orders with volume >= 15)
        filtered_asks = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
        filtered_bids = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]

        mm_ask = min(filtered_asks) if filtered_asks else best_ask
        mm_bid = max(filtered_bids) if filtered_bids else best_bid
        mm_mid_price = (mm_ask + mm_bid) / 2

        if product in ["KELP", "CROISSANTS", "JAMS", "DJEMBES"]:
            # More dynamic strategy for volatile products
            price_history = getattr(self, f"{product.lower()}_prices")
            vwap_history = getattr(self, f"{product.lower()}_vwap")
            
            price_history.append(mm_mid_price)
            if len(price_history) > self.timespan:
                price_history.pop(0)
            
            # Identify market maker orders (volume >= 15)
            mm_asks = {price: volume for price, volume in order_depth.sell_orders.items() if abs(volume) >= 15}
            mm_bids = {price: volume for price, volume in order_depth.buy_orders.items() if abs(volume) >= 15}
            
            # Calculate VWAP of market maker orders
            if mm_asks and mm_bids:
                total_volume = sum(abs(vol) for vol in mm_asks.values()) + sum(abs(vol) for vol in mm_bids.values())
                mm_vwap = sum(price * abs(vol) for price, vol in mm_asks.items()) + sum(price * abs(vol) for price, vol in mm_bids.items())
                mm_vwap /= total_volume
                fair_value = mm_vwap
            else:
                # Fallback to previous method if no market maker orders detected
                volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
                vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume

                vwap_history.append({"vol": volume, "vwap": vwap})
                if len(vwap_history) > self.timespan:
                    vwap_history.pop(0)

                # Use recent VWAP for fair value calculation
                if len(vwap_history) > 0:
                    recent_vwaps = vwap_history[-5:]  # Focus on recent prices
                    total_vol = sum([x["vol"] for x in recent_vwaps])
                    if total_vol > 0:
                        fair_value = sum([x["vwap"] * x["vol"] for x in recent_vwaps]) / total_vol
                    else:
                        fair_value = mm_mid_price
                else:
                    fair_value = mm_mid_price

            take_width = self.take_width[product]
            make_width = self.make_width[product]

        elif product == "RAINFOREST_RESIN":
            # Stable product - use fixed fair value with tight spreads
            fair_value = 10000
            take_width = self.take_width["RAINFOREST_RESIN"]
            make_width = self.make_width["RAINFOREST_RESIN"]

            # Only take orders if significantly off fair value
            if abs(mm_mid_price - fair_value) > 5:
                take_width = take_width * 2  # More aggressive taking when price deviates significantly

        # Taking strategy: take favorable orders
        if best_ask <= fair_value - take_width:
            ask_amount = -1 * order_depth.sell_orders[best_ask]
            if ask_amount <= 20:  # Only take small orders to avoid manipulation
                quantity = min(ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if best_bid >= fair_value + take_width:
            bid_amount = order_depth.buy_orders[best_bid]
            if bid_amount <= 20:  # Only take small orders to avoid manipulation
                quantity = min(bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity

        # Try to clear position near fair value
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            product,
            buy_order_volume,
            sell_order_volume,
            fair_value,
            2,
        )

        # Market making strategy with product-specific spreads
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else int(fair_value + make_width)
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else int(fair_value - make_width)

        # Place buy order
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            buy_price = int(best_bid_below_fair + 1)
            orders.append(Order(product, buy_price, buy_quantity))

        # Place sell order
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            sell_price = int(best_ask_above_fair - 1)
            orders.append(Order(product, sell_price, -sell_quantity))

        return orders

    def close_position(self, product: str, order_depth: OrderDepth, position: int) -> list[Order]:
        """
        Attempt to close an existing position for an inactive product.
        Uses market orders to close the position quickly.
        """
        orders = []
        
        if position == 0:
            return orders
        
        if position > 0:  # We need to sell
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                sell_quantity = min(position, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
        else:  # We need to buy
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                buy_quantity = min(-position, -order_depth.sell_orders[best_ask])
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
        
        return orders

    def squid_ink_strategy(self, order_depth: OrderDepth, position: int, state_timestamp: int) -> list[Order]:
        """
        Special strategy for SQUID_INK focused on mean reversion during high volatility.
        Avoids carrying positions during normal market conditions.
        """
        orders = []
        position_limit = self.position_limits["SQUID_INK"]
        
        # Calculate current mid price
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Update price history
        self.squid_ink_prices.append(mid_price)
        if len(self.squid_ink_prices) > max(self.timespan, self.squid_ink_mean_window):
            self.squid_ink_prices.pop(0)
            
        # Not enough data to make decisions
        if len(self.squid_ink_prices) < 10:
            return orders
            
        # Calculate mean and volatility
        recent_window = min(len(self.squid_ink_prices), self.squid_ink_mean_window)
        mean_price = sum(self.squid_ink_prices[-recent_window:]) / recent_window
        
        # Calculate volatility (standard deviation of recent prices)
        if len(self.squid_ink_prices) >= 2:
            volatility = statistics.stdev(self.squid_ink_prices[-min(10, len(self.squid_ink_prices)):])
        else:
            volatility = 0
            
        # Calculate deviation from mean as a percentage
        deviation_pct = abs(mid_price - mean_price) / mean_price if mean_price > 0 else 0
        
        # First priority: Close existing position if we've held it too long
        if position != 0 and self.squid_ink_position_start_time > 0:
            time_in_position = state_timestamp - self.squid_ink_position_start_time
            if time_in_position >= self.squid_ink_max_position_time:
                return self.close_position("SQUID_INK", order_depth, position)
                
        # If we have no position, look for new mean reversion opportunities
        if position == 0:
            # Only take positions during high volatility AND significant deviation from mean
            if volatility > self.squid_ink_volatility_threshold and deviation_pct > self.squid_ink_deviation_threshold:
                # Price is significantly above mean - take a short position
                if mid_price > mean_price:
                    # Market is overpriced - sell at the bid
                    quantity = min(order_depth.buy_orders[best_bid], position_limit)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -quantity))
                        self.squid_ink_position_start_time = state_timestamp
                        self.squid_ink_last_position = -quantity
                # Price is significantly below mean - take a long position
                elif mid_price < mean_price:
                    # Market is underpriced - buy at the ask
                    quantity = min(-order_depth.sell_orders[best_ask], position_limit)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        self.squid_ink_position_start_time = state_timestamp
                        self.squid_ink_last_position = quantity
        # If we have a position, look for exit opportunities
        else:
            # Exit when price moves back toward mean
            if position > 0:  # We're long
                if mid_price >= mean_price:  # Price has reverted upward
                    # Sell our position at the bid
                    quantity = min(position, order_depth.buy_orders[best_bid])
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -quantity))
                        if quantity == position:  # Full exit
                            self.squid_ink_position_start_time = 0
                            self.squid_ink_last_position = 0
            else:  # We're short
                if mid_price <= mean_price:  # Price has reverted downward
                    # Buy back our position at the ask
                    quantity = min(-position, -order_depth.sell_orders[best_ask])
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        if quantity == -position:  # Full exit
                            self.squid_ink_position_start_time = 0
                            self.squid_ink_last_position = 0
                            
        return orders

    def calculate_synthetic_value(self, state: TradingState, basket_type: str) -> float:
        """Calculate synthetic value of picnic basket based on component prices."""
        if basket_type == "PICNIC_BASKET1":
            # 6 CROISSANTS + 3 JAMS + 1 DJEMBES
            croissant_price = self.calculate_fair_value(state.order_depths["CROISSANTS"])
            jams_price = self.calculate_fair_value(state.order_depths["JAMS"])
            djembes_price = self.calculate_fair_value(state.order_depths["DJEMBES"])
            
            if None in [croissant_price, jams_price, djembes_price]:
                return None
                
            return 6 * croissant_price + 3 * jams_price + 1 * djembes_price
            
        elif basket_type == "PICNIC_BASKET2":
            # 4 CROISSANTS + 2 JAMS
            croissant_price = self.calculate_fair_value(state.order_depths["CROISSANTS"])
            jams_price = self.calculate_fair_value(state.order_depths["JAMS"])
            
            if None in [croissant_price, jams_price]:
                return None
                
            return 4 * croissant_price + 2 * jams_price
            
        return None

    def trade_basket_divergence(
        self, 
        product: str, 
        order_depth: OrderDepth, 
        position: int, 
        synthetic_value: float
    ) -> list[Order]:
        """Trade based on divergence between synthetic value and current price."""
        orders = []
        position_limit = self.position_limits[product]
        
        if synthetic_value is None:
            return orders
            
        # Calculate current basket price
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            current_price = (best_ask + best_bid) / 2
            
            # Calculate divergence
            divergence = synthetic_value - current_price
            
            # Trading thresholds based on basket type
            if product == "PICNIC_BASKET1":
                buy_threshold = 10.0  # Buy when basket is underpriced by 10
                sell_threshold = -10.0  # Sell when basket is overpriced by 10
            else:  # PICNIC_BASKET2
                buy_threshold = 15.0  # Buy when basket is underpriced by 15
                sell_threshold = -15.0  # Sell when basket is overpriced by 15
            
            # Take orders based on divergence
            if divergence > buy_threshold:
                # Basket is underpriced, buy it
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                quantity = min(ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    
            elif divergence < sell_threshold:
                # Basket is overpriced, sell it
                bid_amount = order_depth.buy_orders[best_bid]
                quantity = min(bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    
            # Market making with divergence adjustment
            make_width = 2.0  # Base spread
            if abs(divergence) > 2.0:
                make_width *= 1.5  # Widen spread when divergence is large
                
            # Place buy order
            buy_price = int(current_price - make_width)
            buy_quantity = position_limit - position
            if buy_quantity > 0:
                orders.append(Order(product, buy_price, buy_quantity))
                
            # Place sell order
            sell_price = int(current_price + make_width)
            sell_quantity = position_limit + position
            if sell_quantity > 0:
                orders.append(Order(product, sell_price, -sell_quantity))
                
        return orders

    def hedge_basket_position(
        self,
        state: TradingState,
        basket_type: str,
        basket_position: int
    ) -> dict[str, list[Order]]:
        """Generate hedging orders for basket components based on basket position."""
        orders = {}
        
        # Only hedge 50% of the position
        hedge_position = basket_position // 2
        
        if basket_type == "PICNIC_BASKET1":
            # For each PICNIC_BASKET1, we need to hedge:
            # -6 CROISSANTS
            # -3 JAMS
            # -1 DJEMBES
            target_positions = {
                "CROISSANTS": -6 * hedge_position,
                "JAMS": -3 * hedge_position,
                "DJEMBES": -1 * hedge_position
            }
        else:  # PICNIC_BASKET2
            # For each PICNIC_BASKET2, we need to hedge:
            # -4 CROISSANTS
            # -2 JAMS
            target_positions = {
                "CROISSANTS": -4 * hedge_position,
                "JAMS": -2 * hedge_position
            }
        
        # Generate hedging orders for each component
        for product, target_position in target_positions.items():
            if product in state.order_depths:
                current_position = state.position.get(product, 0)
                position_diff = target_position - current_position
                
                if position_diff != 0:
                    order_depth = state.order_depths[product]
                    fair_value = self.calculate_fair_value(order_depth)
                    
                    if fair_value is not None:
                        if product not in orders:
                            orders[product] = []
                            
                        # Check if we can place orders without exceeding position limits
                        if position_diff > 0:  # Need to buy
                            # Check if buying would exceed position limit
                            if current_position + position_diff <= self.position_limits[product]:
                                # Place buy order at fair value
                                orders[product].append(Order(product, int(fair_value), position_diff))
                        else:  # Need to sell
                            # Check if selling would exceed position limit
                            if current_position + position_diff >= -self.position_limits[product]:
                                # Place sell order at fair value
                                orders[product].append(Order(product, int(fair_value), position_diff))
        
        return orders

    def get_synthetic_basket_order_depth(
        self, 
        state: TradingState, 
        basket_type: str
    ) -> OrderDepth:
        """Calculate synthetic order depth for a basket based on its components."""
        # Initialize the synthetic basket order depth
        synthetic_order_depth = OrderDepth()
        
        # Define basket components and weights
        if basket_type == "PICNIC_BASKET1":
            components = {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            }
        else:  # PICNIC_BASKET2
            components = {
                "CROISSANTS": 4,
                "JAMS": 2
            }
        
        # Calculate best bids and asks for each component
        component_bids = {}
        component_asks = {}
        for product, weight in components.items():
            if product in state.order_depths and state.order_depths[product].buy_orders:
                component_bids[product] = max(state.order_depths[product].buy_orders.keys())
            else:
                component_bids[product] = 0
                
            if product in state.order_depths and state.order_depths[product].sell_orders:
                component_asks[product] = min(state.order_depths[product].sell_orders.keys())
            else:
                component_asks[product] = float('inf')
        
        # Calculate implied bid (what you could sell basket for by buying components)
        implied_bid = sum(component_bids[product] * weight for product, weight in components.items())
        
        # Calculate implied ask (what you could buy basket for by selling components)
        implied_ask = sum(component_asks[product] * weight for product, weight in components.items())
        
        # Calculate maximum number of baskets that could be created/disassembled
        if implied_bid > 0:
            bid_volumes = []
            for product, weight in components.items():
                if product in state.order_depths and component_bids[product] > 0:
                    # How many baskets can be created based on this component's available volume
                    volume = state.order_depths[product].buy_orders[component_bids[product]] // weight
                    bid_volumes.append(volume)
                else:
                    bid_volumes.append(0)
            
            implied_bid_volume = min(bid_volumes) if bid_volumes else 0
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
        
        if implied_ask < float('inf'):
            ask_volumes = []
            for product, weight in components.items():
                if product in state.order_depths and component_asks[product] < float('inf'):
                    # How many baskets can be disassembled based on this component's available volume
                    volume = abs(state.order_depths[product].sell_orders[component_asks[product]]) // weight
                    ask_volumes.append(volume)
                else:
                    ask_volumes.append(0)
            
            implied_ask_volume = min(ask_volumes) if ask_volumes else 0
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_depth
    
    def execute_basket_arbitrage(
        self, 
        state: TradingState, 
        basket_type: str
    ) -> dict[str, list[Order]]:
        """Execute arbitrage between basket and its components when profitable."""
        result = {}
        
        # Get the actual basket order depth
        basket_order_depth = state.order_depths[basket_type]
        
        # Get the synthetic basket order depth
        synthetic_order_depth = self.get_synthetic_basket_order_depth(state, basket_type)
        
        # Define basket components and weights
        if basket_type == "PICNIC_BASKET1":
            components = {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            }
        else:  # PICNIC_BASKET2
            components = {
                "CROISSANTS": 4,
                "JAMS": 2
            }
        
        # Calculate arbitrage opportunities
        
        # Opportunity 1: Buy the basket, sell the components
        if basket_order_depth.sell_orders and synthetic_order_depth.buy_orders:
            basket_ask = min(basket_order_depth.sell_orders.keys())
            synthetic_bid = max(synthetic_order_depth.buy_orders.keys())
            
            # If buying basket and selling components is profitable
            if basket_ask < synthetic_bid:
                basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask])
                synthetic_bid_volume = synthetic_order_depth.buy_orders[synthetic_bid]
                
                # How many baskets to arbitrage
                arb_volume = min(basket_ask_volume, synthetic_bid_volume)
                
                # Limit by position limits
                basket_position = state.position.get(basket_type, 0)
                arb_volume = min(arb_volume, self.position_limits[basket_type] - basket_position)
                
                if arb_volume > 0:
                    # Buy the basket
                    if basket_type not in result:
                        result[basket_type] = []
                    result[basket_type].append(Order(basket_type, basket_ask, arb_volume))
                    
                    # Sell the components
                    for product, weight in components.items():
                        if product not in result:
                            result[product] = []
                        
                        # Find best bid for this component
                        if product in state.order_depths and state.order_depths[product].buy_orders:
                            best_bid = max(state.order_depths[product].buy_orders.keys())
                            result[product].append(Order(product, best_bid, -weight * arb_volume))
        
        # Opportunity 2: Buy the components, sell the basket
        if basket_order_depth.buy_orders and synthetic_order_depth.sell_orders:
            basket_bid = max(basket_order_depth.buy_orders.keys())
            synthetic_ask = min(synthetic_order_depth.sell_orders.keys())
            
            # If buying components and selling basket is profitable
            if basket_bid > synthetic_ask:
                basket_bid_volume = basket_order_depth.buy_orders[basket_bid]
                synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask])
                
                # How many baskets to arbitrage
                arb_volume = min(basket_bid_volume, synthetic_ask_volume)
                
                # Limit by position limits
                basket_position = state.position.get(basket_type, 0)
                arb_volume = min(arb_volume, self.position_limits[basket_type] + basket_position)
                
                if arb_volume > 0:
                    # Sell the basket
                    if basket_type not in result:
                        result[basket_type] = []
                    result[basket_type].append(Order(basket_type, basket_bid, -arb_volume))
                    
                    # Buy the components
                    for product, weight in components.items():
                        if product not in result:
                            result[product] = []
                        
                        # Find best ask for this component
                        if product in state.order_depths and state.order_depths[product].sell_orders:
                            best_ask = min(state.order_depths[product].sell_orders.keys())
                            result[product].append(Order(product, best_ask, weight * arb_volume))
        
        return result

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Main method required by the competition.
        It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}

        # Load saved state if available
        if state.traderData and state.traderData != "SAMPLE":
            try:
                trader_data = jsonpickle.decode(state.traderData)
                # Load all price histories
                for product in ["kelp", "resin", "squid_ink", "croissants", "jams", "djembes"]:
                    prices_key = f"{product}_prices"
                    vwap_key = f"{product}_vwap"
                    if prices_key in trader_data:
                        setattr(self, prices_key, trader_data[prices_key])
                    if vwap_key in trader_data:
                        setattr(self, vwap_key, trader_data[vwap_key])
            except:
                logger.print("Could not parse trader data")

        # Process each product
        for product in state.order_depths.keys():
            if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                # Try to execute basket arbitrage
                arbitrage_orders = self.execute_basket_arbitrage(state, product)
                
                # If no arbitrage was possible, fall back to divergence trading
                if not arbitrage_orders or product not in arbitrage_orders:
                    # Calculate synthetic value and trade divergence
                    synthetic_value = self.calculate_synthetic_value(state, product)
                    position = state.position.get(product, 0)
                    orders = self.trade_basket_divergence(product, state.order_depths[product], position, synthetic_value)
                    if orders:
                        result[product] = orders
                else:
                    # Merge arbitrage orders into result
                    for p, orders in arbitrage_orders.items():
                        if p in result:
                            result[p].extend(orders)
                        else:
                            result[p] = orders
                    
                # Only hedge if we couldn't execute a clean arbitrage
                if product not in arbitrage_orders:
                    # Generate hedging orders for basket components
                    position = state.position.get(product, 0)
                    hedge_orders = self.hedge_basket_position(state, product, position)
                    for component, component_orders in hedge_orders.items():
                        if component in result:
                            result[component].extend(component_orders)
                        else:
                            result[component] = component_orders
                        
            elif product in self.active_products and self.active_products[product]:
                position = state.position.get(product, 0)
                orders = self.product_orders(product, state.order_depths[product], position, state.timestamp)
                if orders:
                    result[product] = orders
            else:
                # For inactive products, if we have a position, try to close it
                position = state.position.get(product, 0)
                if position != 0:
                    orders = self.close_position(product, state.order_depths[product], position)
                    if orders:
                        result[product] = orders

        # Save state for next iteration
        trader_data = {
            "kelp_prices": self.kelp_prices,
            "resin_prices": self.resin_prices,
            "squid_ink_prices": self.squid_ink_prices,
            "croissants_prices": self.croissants_prices,
            "jams_prices": self.jams_prices,
            "djembes_prices": self.djembes_prices,
            "kelp_vwap": self.kelp_vwap,
            "resin_vwap": self.resin_vwap,
            "squid_ink_vwap": self.squid_ink_vwap,
            "croissants_vwap": self.croissants_vwap,
            "jams_vwap": self.jams_vwap,
            "djembes_vwap": self.djembes_vwap,
        }

        serialized_trader_data = jsonpickle.encode(trader_data)
        conversions = 0  # No conversions in this strategy

        logger.flush(state, result, conversions, serialized_trader_data)

        return result, conversions, serialized_trader_data 