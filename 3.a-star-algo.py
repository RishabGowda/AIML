import numpy as np, heapq

def solve_puzzle(initial, goal):
    pq, seen = [(0, initial.tolist(), [])], set()
    while pq:
        _, state, path = heapq.heappop(pq)
        if state == goal.tolist(): return path + [state]
        if str(state) in seen: continue
        seen.add(str(state))
        zero = tuple(map(int, np.where(np.array(state) == 0)))
        for d in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = zero[0] + d[0], zero[1] + d[1]
            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
                new_state = np.array(state)
                new_state[zero], new_state[new_pos] = new_state[new_pos], new_state[zero]
                heapq.heappush(pq, (np.sum(new_state != goal) + len(path), new_state.tolist(), path + [state]))

initial = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])
goal = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
solution = solve_puzzle(initial, goal)

for i, move in enumerate(solution): print(f"Move {i}:\n{np.array(move)}\n")
print("Total moves:", len(solution) - 1)
