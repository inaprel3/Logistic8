import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Store coordinates (Appendix 2)
stores = {
    1: (19, 9), 2: (25, 6), 3: (28, 4), 4: (27, 2), 5: (20, 5),
    6: (18, 3), 7: (16, 7), 8: (13, 3), 9: (9, 2), 10: (11, 7),
    11: (4, 4), 12: (6, 7), 13: (2, 8), 14: (12, 9), 15: (4, 11),
    16: (8, 12), 17: (2, 14), 18: (8, 15), 19: (13, 12), 20: (12, 15),
    21: (15, 14), 22: (16, 17), 23: (18, 12), 24: (20, 16), 25: (23, 17),
    26: (23, 14), 27: (27, 16), 28: (30, 15), 29: (24, 10), 30: (28, 8)
}
warehouse = (16, 10)

# Orders (Appendix 3) for Monday, Tuesday, Thursday
orders = {
    'Monday': {
        1: {'П': 0, 'М': 10, 'Н': 8}, 2: {'П': 20, 'М': 26, 'Н': 18}, 3: {'П': 44, 'М': 24, 'Н': 26},
        4: {'П': 10, 'М': 10, 'Н': 18}, 5: {'П': 26, 'М': 34, 'Н': 20}, 6: {'П': 32, 'М': 20, 'Н': 0},
        7: {'П': 20, 'М': 8, 'Н': 0}, 8: {'П': 20, 'М': 14, 'Н': 24}, 9: {'П': 28, 'М': 10, 'Н': 6},
        10: {'П': 40, 'М': 20, 'Н': 12}, 11: {'П': 44, 'М': 20, 'Н': 20}, 12: {'П': 24, 'М': 8, 'Н': 6},
        13: {'П': 30, 'М': 20, 'Н': 36}, 14: {'П': 20, 'М': 10, 'Н': 0}, 15: {'П': 16, 'М': 6, 'Н': 10},
        16: {'П': 10, 'М': 4, 'Н': 6}, 17: {'П': 46, 'М': 0, 'Н': 32}, 18: {'П': 14, 'М': 6, 'Н': 20},
        19: {'П': 12, 'М': 8, 'Н': 0}, 20: {'П': 24, 'М': 8, 'Н': 0}, 21: {'П': 0, 'М': 0, 'Н': 40},
        22: {'П': 20, 'М': 8, 'Н': 12}, 23: {'П': 10, 'М': 0, 'Н': 0}, 24: {'П': 10, 'М': 0, 'Н': 0},
        25: {'П': 14, 'М': 4, 'Н': 16}, 26: {'П': 34, 'М': 24, 'Н': 20}, 27: {'П': 30, 'М': 0, 'Н': 14},
        28: {'П': 20, 'М': 16, 'Н': 20}, 29: {'П': 16, 'М': 32, 'Н': 12}, 30: {'П': 24, 'М': 16, 'Н': 20}
    },
    'Tuesday': {
        1: {'П': 0, 'М': 0, 'Н': 16}, 2: {'П': 24, 'М': 16, 'Н': 0}, 3: {'П': 48, 'М': 16, 'Н': 38},
        4: {'П': 0, 'М': 0, 'Н': 16}, 5: {'П': 40, 'М': 24, 'Н': 20}, 6: {'П': 30, 'М': 10, 'Н': 50},
        7: {'П': 34, 'М': 8, 'Н': 10}, 8: {'П': 20, 'М': 8, 'Н': 0}, 9: {'П': 0, 'М': 0, 'Н': 20},
        10: {'П': 40, 'М': 0, 'Н': 16}, 11: {'П': 28, 'М': 12, 'Н': 24}, 12: {'П': 20, 'М': 0, 'Н': 5},
        13: {'П': 18, 'М': 10, 'Н': 14}, 14: {'П': 0, 'М': 0, 'Н': 10}, 15: {'П': 12, 'М': 12, 'Н': 15},
        16: {'П': 20, 'М': 0, 'Н': 10}, 17: {'П': 18, 'М': 16, 'Н': 0}, 18: {'П': 28, 'М': 5, 'Н': 32},
        19: {'П': 10, 'М': 8, 'Н': 16}, 20: {'П': 0, 'М': 10, 'Н': 12}, 21: {'П': 24, 'М': 20, 'Н': 0},
        22: {'П': 12, 'М': 8, 'Н': 14}, 23: {'П': 20, 'М': 16, 'Н': 24}, 24: {'П': 50, 'М': 20, 'Н': 32},
        25: {'П': 14, 'М': 10, 'Н': 16}, 26: {'П': 20, 'М': 5, 'Н': 12}, 27: {'П': 46, 'М': 32, 'Н': 42},
        28: {'П': 20, 'М': 16, 'Н': 0}, 29: {'П': 16, 'М': 12, 'Н': 6}, 30: {'П': 26, 'М': 6, 'Н': 12}
    },
    'Thursday': {
        1: {'П': 4, 'М': 0, 'Н': 32}, 2: {'П': 20, 'М': 8, 'Н': 0}, 3: {'П': 20, 'М': 10, 'Н': 10},
        4: {'П': 50, 'М': 8, 'Н': 12}, 5: {'П': 50, 'М': 10, 'Н': 30}, 6: {'П': 35, 'М': 10, 'Н': 22},
        7: {'П': 16, 'М': 14, 'Н': 12}, 8: {'П': 10, 'М': 4, 'Н': 10}, 9: {'П': 40, 'М': 10, 'Н': 13},
        10: {'П': 0, 'М': 0, 'Н': 22}, 11: {'П': 0, 'М': 0, 'Н': 0}, 12: {'П': 20, 'М': 12, 'Н': 10},
        13: {'П': 16, 'М': 10, 'Н': 28}, 14: {'П': 0, 'М': 0, 'Н': 0}, 15: {'П': 35, 'М': 18, 'Н': 32},
        16: {'П': 0, 'М': 0, 'Н': 0}, 17: {'П': 44, 'М': 32, 'Н': 32}, 18: {'П': 0, 'М': 0, 'Н': 0},
        19: {'П': 0, 'М': 0, 'Н': 0}, 20: {'П': 30, 'М': 0, 'Н': 16}, 21: {'П': 40, 'М': 20, 'Н': 50},
        22: {'П': 12, 'М': 0, 'Н': 26}, 23: {'П': 24, 'М': 0, 'Н': 10}, 24: {'П': 10, 'М': 0, 'Н': 16},
        25: {'П': 14, 'М': 10, 'Н': 20}, 26: {'П': 0, 'М': 0, 'Н': 0}, 27: {'П': 41, 'М': 0, 'Н': 42},
        28: {'П': 40, 'М': 40, 'Н': 45}, 29: {'П': 32, 'М': 8, 'Н': 0}, 30: {'П': 44, 'М': 0, 'Н': 16}
    }
}

# Constants
VEHICLE_CAPACITY = 120
OWN_VEHICLES = 6
SPEED = 20  # km/h, 1 km = 3 min
LOADING_TIME = 30  # min for subsequent trips
UNLOADING_TIME_PER_BOX = 0.5  # min
STORE_OPERATIONS_TIME = 15  # min per store
BREAK_THRESHOLD = 110  # km (5.5 hours)
BREAK_TIME = 30  # min
MAX_WORK_HOURS = 11
MIN_WORK_HOURS = 6
OWN_FIXED_COST = 300  # UAH/day
OWN_VARIABLE_COST = 15  # UAH/km
HIRED_FIXED_COST = 1500  # UAH/day
HIRED_VARIABLE_COST = 30  # UAH/km
OVERTIME_RATE = 5  # UAH/min
MIN_LOAD = 90
UNDERLOAD_PENALTY = 50  # UAH per box below 90
OWN_UNDERWORK_PENALTY = 300  # UAH/day
HIRED_UNDERWORK_PENALTY = 500  # UAH/day
GUARD_COST = 600  # UAH/day per hired vehicle with drinks
UNUSED_OWN_VEHICLE_COST = 1500  # UAH/day
DELAY_PENALTY = 100  # UAH/box/day

def manhattan_distance(p1, p2):
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_route_distance(route, stores, warehouse):
    """Calculate total Manhattan distance for a route starting and ending at warehouse."""
    if not route:
        return 0
    distance = manhattan_distance(warehouse, stores[route[0]])
    for i in range(len(route) - 1):
        distance += manhattan_distance(stores[route[i]], stores[route[i + 1]])
    distance += manhattan_distance(stores[route[-1]], warehouse)
    return distance

def calculate_route_time(route, load, stores, warehouse, is_first_trip):
    """Calculate total time for a route in minutes."""
    distance = calculate_route_distance(route, stores, warehouse)
    travel_time = distance * 3  # 3 min/km
    unloading_time = load * UNLOADING_TIME_PER_BOX
    operations_time = len(route) * STORE_OPERATIONS_TIME
    break_time = BREAK_TIME if distance > BREAK_THRESHOLD else 0
    loading_time = 0 if is_first_trip else LOADING_TIME
    return travel_time + unloading_time + operations_time + break_time + loading_time

def calculate_route_cost(route, load, vehicle_type, stores, warehouse, has_drinks):
    """Calculate cost for a route."""
    distance = calculate_route_distance(route, stores, warehouse)
    fixed_cost = OWN_FIXED_COST if vehicle_type == 'own' else HIRED_FIXED_COST
    variable_cost = OWN_VARIABLE_COST if vehicle_type == 'own' else HIRED_VARIABLE_COST
    route_cost = variable_cost * distance
    guard_cost = GUARD_COST if vehicle_type == 'hired' and has_drinks else 0
    underload_penalty = UNDERLOAD_PENALTY * (MIN_LOAD - load) if load < MIN_LOAD and route else 0
    return fixed_cost, route_cost, guard_cost, underload_penalty

def create_routes(day, orders, stores, warehouse):
    """Create routes for a given day using a heuristic approach."""
    remaining_orders = {store: order.copy() for store, order in orders[day].items()}
    routes = []
    route_types = []  # 'П+Н' or 'М+Н'
    loads = []
    distances = []
    times = []
    
    def get_angle(store_id):
        x, y = stores[store_id]
        wx, wy = warehouse
        return math.atan2(y - wy, x - wx)
    
    # Sort stores by angle from warehouse
    sorted_stores = sorted(stores.keys(), key=get_angle)
    
    while any(sum(order.values()) > 0 for order in remaining_orders.values()):
        route = []
        load = 0
        route_type = None
        has_drinks = False
        
        for store_id in sorted_stores:
            order = remaining_orders[store_id]
            if sum(order.values()) == 0:
                continue
                
            # Try П+Н
            if route_type is None or route_type == 'П+Н':
                new_load = load + order['П'] + order['Н']
                if new_load <= VEHICLE_CAPACITY and (order['П'] > 0 or order['Н'] > 0):
                    route.append(store_id)
                    load = new_load
                    route_type = 'П+Н'
                    has_drinks = has_drinks or order['Н'] > 0
                    remaining_orders[store_id]['П'] = 0
                    remaining_orders[store_id]['Н'] = 0
                    if order['М'] == 0:  # Clear store if no М remaining
                        remaining_orders[store_id]['М'] = 0
                    continue
            
            # Try М+Н
            if route_type is None or route_type == 'М+Н':
                new_load = load + order['М'] + order['Н']
                if new_load <= VEHICLE_CAPACITY and (order['М'] > 0 or order['Н'] > 0):
                    route.append(store_id)
                    load = new_load
                    route_type = 'М+Н'
                    has_drinks = has_drinks or order['Н'] > 0
                    remaining_orders[store_id]['М'] = 0
                    remaining_orders[store_id]['Н'] = 0
                    if order['П'] == 0:  # Clear store if no П remaining
                        remaining_orders[store_id]['П'] = 0
                    continue
        
        if route:
            routes.append(route)
            route_types.append(route_type)
            loads.append(load)
            distances.append(calculate_route_distance(route, stores, warehouse))
            times.append(calculate_route_time(route, load, stores, warehouse, len(routes) == 1))
    
    return routes, route_types, loads, distances, times

def schedule_routes(routes, times, loads, route_types):
    """Schedule routes to vehicles, prioritizing own vehicles."""
    schedule = []
    vehicle_times = []
    vehicle_types = []
    current_time = datetime.strptime("08:00", "%H:%M")
    own_vehicles_used = 0
    hired_vehicles_used = 0
    
    for i, (route, time, load, r_type) in enumerate(zip(routes, times, loads, route_types)):
        vehicle_type = 'own' if own_vehicles_used < OWN_VEHICLES else 'hired'
        if vehicle_type == 'own':
            own_vehicles_used += 1
        else:
            hired_vehicles_used += 1
        
        arrival_time = current_time + timedelta(minutes=time)
        schedule.append({
            'vehicle': len(schedule) + 1,
            'route': i + 1,
            'departure': current_time.strftime("%H:%M"),
            'arrival': arrival_time.strftime("%H:%M"),
            'time': time,
            'type': vehicle_type,
            'load': load,
            'has_drinks': 'Н' in r_type
        })
        vehicle_times.append(time)
        vehicle_types.append(vehicle_type)
        current_time = arrival_time + timedelta(minutes=30)  # Loading time for next trip
    
    return schedule, vehicle_times, vehicle_types

def calculate_costs(schedule, routes, loads, distances, route_types, stores, warehouse, own_vehicles_used):
    """Calculate total costs including penalties."""
    total_cost = 0
    total_load = sum(loads)
    total_distance = sum(distances)
    underload_penalty = 0
    underwork_penalty = 0
    overtime_cost = 0
    guard_cost = 0
    vehicle_costs = []
    
    for entry in schedule:
        route_idx = entry['route'] - 1
        vehicle_type = entry['type']
        load = entry['load']
        distance = distances[route_idx]
        has_drinks = entry['has_drinks']
        
        fixed_cost, route_cost, route_guard_cost, route_underload_penalty = calculate_route_cost(
            routes[route_idx], load, vehicle_type, stores, warehouse, has_drinks
        )
        guard_cost += route_guard_cost
        underload_penalty += route_underload_penalty if route_idx != len(routes) - 1 else 0  # Last route exempt
        
        time_hours = entry['time'] / 60
        if time_hours < MIN_WORK_HOURS:
            underwork_penalty += OWN_UNDERWORK_PENALTY if vehicle_type == 'own' else HIRED_UNDERWORK_PENALTY
        if time_hours > 8:
            overtime_cost += (time_hours - 8) * 60 * OVERTIME_RATE
        
        vehicle_costs.append(fixed_cost + route_cost + route_guard_cost + route_underload_penalty)
    
    # Penalty for unused own vehicles
    unused_vehicles = max(0, OWN_VEHICLES - own_vehicles_used)
    unused_cost = unused_vehicles * UNUSED_OWN_VEHICLE_COST
    
    total_cost = sum(vehicle_costs) + underload_penalty + underwork_penalty + overtime_cost + guard_cost + unused_cost
    return total_cost, total_load, total_distance, underload_penalty, underwork_penalty, overtime_cost, guard_cost, unused_cost

def analyze_delivery(total_cost, total_load, total_distance, num_routes):
    """Perform analysis as per Appendix 8."""
    capacity_utilization = total_load / (num_routes * VEHICLE_CAPACITY) if num_routes > 0 else 0
    cost_per_km = total_cost / total_distance if total_distance > 0 else 0
    cost_per_unit = total_cost / total_load if total_load > 0 else 0
    return {
        'total_cost': total_cost,
        'total_load': total_load,
        'total_distance': total_distance,
        'num_routes': num_routes,
        'capacity_utilization': capacity_utilization,
        'cost_per_km': cost_per_km,
        'cost_per_unit': cost_per_unit
    }

def plot_routes(day, routes, stores, warehouse):
    """Plot routes for a given day."""
    plt.figure(figsize=(10, 8))
    # Plot warehouse
    plt.scatter(warehouse[0], warehouse[1], c='red', marker='s', s=200, label='Склад')
    # Plot stores
    for store_id, (x, y) in stores.items():
        plt.scatter(x, y, c='blue', marker='o', s=50)
        plt.text(x + 0.5, y + 0.5, f'{store_id}', fontsize=9)
    
    # Plot routes
    colors = ['green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'pink', 'brown']
    for i, route in enumerate(routes):
        path = [warehouse] + [stores[store_id] for store_id in route] + [warehouse]
        x, y = zip(*path)
        plt.plot(x, y, c=colors[i % len(colors)], label=f'Маршрут {i + 1}', linewidth=2)
    
    plt.title(f'Маршрути доставки: {day}')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'routes_{day.lower()}.png')
    plt.close()

def plot_analysis(days, results):
    """Plot analysis metrics."""
    # Total costs
    plt.figure(figsize=(10, 6))
    costs = [results[day]['total_cost'] for day in days]
    plt.bar(days, costs, color='blue')
    plt.title('Загальні витрати за днями')
    plt.xlabel('День')
    plt.ylabel('Витрати (грн)')
    plt.grid(True, axis='y')
    plt.savefig('costs.png')
    plt.close()
    
    # Capacity utilization
    plt.figure(figsize=(10, 6))
    utilizations = [results[day]['analysis']['capacity_utilization'] for day in days]
    plt.bar(days, utilizations, color='green')
    plt.title('Коефіцієнт вантажомісткості за днями')
    plt.xlabel('День')
    plt.ylabel('Коефіцієнт')
    plt.grid(True, axis='y')
    plt.savefig('capacity_utilization.png')
    plt.close()
    
    # Cost per km and per unit
    plt.figure(figsize=(10, 6))
    cost_per_km = [results[day]['analysis']['cost_per_km'] for day in days]
    cost_per_unit = [results[day]['analysis']['cost_per_unit'] for day in days]
    x = np.arange(len(days))
    plt.bar(x - 0.2, cost_per_km, 0.4, label='Витрати на 1 км (грн)', color='purple')
    plt.bar(x + 0.2, cost_per_unit, 0.4, label='Витрати на одиницю вантажу (грн)', color='orange')
    plt.xticks(x, days)
    plt.title('Витрати на 1 км та на одиницю вантажу')
    plt.xlabel('День')
    plt.ylabel('Витрати (грн)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('cost_metrics.png')
    plt.close()

# Main processing for each day
days = ['Monday', 'Tuesday', 'Thursday']
results = {}

for day in days:
    routes, route_types, loads, distances, times = create_routes(day, orders, stores, warehouse)
    schedule, vehicle_times, vehicle_types = schedule_routes(routes, times, loads, route_types)
    own_vehicles_used = sum(1 for entry in schedule if entry['type'] == 'own')
    total_cost, total_load, total_distance, underload_penalty, underwork_penalty, overtime_cost, guard_cost, unused_cost = calculate_costs(
        schedule, routes, loads, distances, route_types, stores, warehouse, own_vehicles_used
    )
    analysis = analyze_delivery(total_cost, total_load, total_distance, len(routes))
    
    results[day] = {
        'routes': routes,
        'route_types': route_types,
        'loads': loads,
        'distances': distances,
        'times': times,
        'schedule': schedule,
        'total_cost': total_cost,
        'total_load': total_load,
        'total_distance': total_distance,
        'underload_penalty': underload_penalty,
        'underwork_penalty': underwork_penalty,
        'overtime_cost': overtime_cost,
        'guard_cost': guard_cost,
        'unused_cost': unused_cost,
        'analysis': analysis
    }
    
    # Plot routes for the day
    plot_routes(day, routes, stores, warehouse)

# Plot analysis graphs
plot_analysis(days, results)

# Output results in the required formats
def print_route_parameters(day, routes, route_types, loads, distances, times, orders):
    print(f"\n=== Route Parameters for {day} (Appendix 4) ===")
    for i, (route, r_type, load, dist, time) in enumerate(zip(routes, route_types, loads, distances, times)):
        print(f"\nRoute {i + 1}:")
        print("| № марш | № магаз | П | М | Н |")
        print("|--------|---------|---|---|---|")
        for store in route:
            order = orders[day][store]
            p = order['П'] if r_type == 'П+Н' and order['П'] > 0 else 0
            m = order['М'] if r_type == 'М+Н' and order['М'] > 0 else 0
            n = order['Н'] if order['Н'] > 0 else 0
            print(f"| {i + 1} | {store} | {p} | {m} | {n} |")
        print(f"\nResult:")
        print(f"Path: M: 0-{'-'.join(map(str, route))}-0")
        print(f"Distance: L = {dist} km")
        print(f"Load: P = {load} boxes")
        print(f"Time: T = {time} min ({time // 60}h {time % 60}m)")

def print_schedule(day, schedule):
    print(f"\n=== Transport Schedule for {day} (Appendix 5) ===")
    print("| № авто | № маршрута | Відправлення | Прибуття | Загальний час | Належність |")
    print("|--------|------------|--------------|----------|---------------|------------|")
    for entry in schedule:
        hours = entry['time'] // 60
        minutes = entry['time'] % 60
        print(f"| {entry['vehicle']} | {entry['route']} | {entry['departure']} | {entry['arrival']} | {hours}h {minutes}m | {entry['type']} |")

def print_costs(day, schedule, routes, loads, distances, route_types, stores, warehouse, unused_cost):
    print(f"\n=== Cost Calculation for {day} (Appendix 6) ===")
    print("| № авто | Належність | Маршрути | Коробки | Пробіг, км | Плата за пробіг | Витрати авто | Понаднорм | Недовантаження | Недовикористання | Охорона | Всього |")
    print("|--------|------------|----------|---------|------------|-----------------|--------------|-----------|----------------|------------------|---------|--------|")
    total = 0
    for entry in schedule:
        route_idx = entry['route'] - 1
        vehicle_type = entry['type']
        load = loads[route_idx]
        distance = distances[route_idx]
        has_drinks = entry['has_drinks']
        fixed_cost, route_cost, guard_cost, underload_penalty = calculate_route_cost(
            routes[route_idx], load, vehicle_type, stores, warehouse, has_drinks
        )
        time_hours = entry['time'] / 60
        overtime = (time_hours - 8) * 60 * OVERTIME_RATE if time_hours > 8 else 0
        underwork = OWN_UNDERWORK_PENALTY if vehicle_type == 'own' and time_hours < MIN_WORK_HOURS else \
                    HIRED_UNDERWORK_PENALTY if vehicle_type == 'hired' and time_hours < MIN_WORK_HOURS else 0
        vehicle_total = fixed_cost + route_cost + guard_cost + underload_penalty + overtime + underwork
        total += vehicle_total
        print(f"| {entry['vehicle']} | {vehicle_type} | {entry['route']} | {load} | {distance} | {route_cost} | {fixed_cost} | {overtime} | {underload_penalty} | {underwork} | {guard_cost} | {vehicle_total} |")
    total += unused_cost
    print(f"Unused own vehicles cost: {unused_cost} UAH")
    print(f"Total: {total} UAH")
    return total

def print_plan(day, routes, route_types, orders):
    print(f"\n=== Delivery Plan for {day} (Appendix 7) ===")
    print("| № маршруту | № магазину | П | М | Н |")
    print("|------------|------------|---|---|---|")
    for i, (route, r_type) in enumerate(zip(routes, route_types)):
        for store in route:
            order = orders[day][store]
            p = order['П'] if r_type == 'П+Н' and order['П'] > 0 else 0
            m = order['М'] if r_type == 'М+Н' and order['М'] > 0 else 0
            n = order['Н'] if order['Н'] > 0 else 0
            print(f"| {i + 1} | {store} | {p} | {m} | {n} |")

def print_analysis(days, results):
    print("\n=== Delivery Analysis (Appendix 8) ===")
    print("| Показник | Формула | " + " | ".join(days) + " | Всього |")
    print("|----------|---------|" + "----|" * len(days) + "-------|")
    
    total_week_cost = sum(r['total_cost'] for r in results.values())
    total_week_load = sum(r['total_load'] for r in results.values())
    total_week_distance = sum(r['total_distance'] for r in results.values())
    total_week_routes = sum(r['analysis']['num_routes'] for r in results.values())
    
    print(f"| Загальні витрати | Собщ | {' | '.join(str(r['total_cost']) for r in results.values())} | {total_week_cost} |")
    print(f"| Кількість вантажу | Робщ | {' | '.join(str(r['total_load']) for r in results.values())} | {total_week_load} |")
    print(f"| Пробіг авто | Lобщ | {' | '.join(str(r['total_distance']) for r in results.values())} | {total_week_distance} |")
    print(f"| Кількість маршрутів | N | {' | '.join(str(r['analysis']['num_routes']) for r in results.values())} | {total_week_routes} |")
    print(f"| Коеф. вантажомісткості | K=Робщ/(N*Q) | {' | '.join(f'{r['analysis']['capacity_utilization']:.2f}' for r in results.values())} | {(total_week_load / (total_week_routes * VEHICLE_CAPACITY)):.2f} |")
    print(f"| Витрати на 1 км | CL=Собщ/Lобщ | {' | '.join(f'{r['analysis']['cost_per_km']:.2f}' for r in results.values())} | {(total_week_cost / total_week_distance):.2f} |")
    print(f"| Витрати на одиницю вантажу | Cp=Собщ/Робщ | {' | '.join(f'{r['analysis']['cost_per_unit']:.2f}' for r in results.values())} | {(total_week_cost / total_week_load):.2f} |")

# Execute for all days
for day in days:
    print_route_parameters(day, results[day]['routes'], results[day]['route_types'], results[day]['loads'], results[day]['distances'], results[day]['times'], orders)
    print_schedule(day, results[day]['schedule'])
    results[day]['total_cost'] = print_costs(day, results[day]['schedule'], results[day]['routes'], results[day]['loads'], results[day]['distances'], results[day]['route_types'], stores, warehouse, results[day]['unused_cost'])
    print_plan(day, results[day]['routes'], results[day]['route_types'], orders)

print_analysis(days, results)
