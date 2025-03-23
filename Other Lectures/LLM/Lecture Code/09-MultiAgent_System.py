#%%--------------------------------------------------------------------------------------------------------------------
import threading
import queue
import random
import time

def data_collector(data_queue, control_queue, num_data_points=5):
    for _ in range(num_data_points):
        data_point = random.randint(0, 100)
        print(f"[DataCollector] Generated data: {data_point}")
        data_queue.put(data_point)
        time.sleep(0.5)
    data_queue.put("stop")
    control_queue.put("stop")

def data_processor(data_queue, result_queue, control_queue):
    data_sum = 0
    data_count = 0
    while True:
        data = data_queue.get()
        if data == "stop":
            print("[DataProcessor] Stopping.")
            result_queue.put("stop")
            control_queue.put("stop")
            break
        data_sum += data
        data_count += 1
        current_avg = data_sum / data_count
        print(f"[DataProcessor] Current average: {current_avg:.2f}")
        result_queue.put(current_avg)

def result_saver(result_queue, control_queue, results_list):
    while True:
        result = result_queue.get()
        if result == "stop":
            print("[ResultSaver] Stopping.")
            control_queue.put("stop")
            break
        results_list.append(result)
        print(f"[ResultSaver] Saved result: {result:.2f}")

#%%--------------------------------------------------------------------------------------------------------------------
data_queue = queue.Queue()
result_queue = queue.Queue()
control_queue = queue.Queue()
final_results = []

#%%--------------------------------------------------------------------------------------------------------------------
collector_thread = threading.Thread(target=data_collector,args=(data_queue, control_queue, 5), daemon=True)
processor_thread = threading.Thread(target=data_processor,args=(data_queue, result_queue, control_queue),daemon=True)
saver_thread = threading.Thread(target=result_saver,args=(result_queue, control_queue, final_results),daemon=True)

#%%--------------------------------------------------------------------------------------------------------------------
collector_thread.start()
processor_thread.start()
saver_thread.start()
stop_count = 0
while stop_count < 3:
    signal = control_queue.get()
    if signal == "stop":
        stop_count += 1
print("\nAll agents have stopped.")
print("Final saved results:", final_results)
