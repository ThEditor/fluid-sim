#!/bin/bash
#Build
make all

# Configuration
RESULTS_FILE="benchmark_results.csv"
THREAD_COUNTS=(2 4 8)  # Thread counts to test
DURATION=5                 # Test duration in seconds
RUNS_PER_CONFIG=1           # Number of runs per thread count for averaging

# Ensure we're in the out directory
cd out

# Create or overwrite results file with header
echo "Threads,Run,FPS,CPU_Usage" > $RESULTS_FILE

# Run benchmarks
for threads in "${THREAD_COUNTS[@]}"; do
    echo "===================================="
    echo "Testing with $threads threads"
    echo "===================================="
    
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        echo "Run $run of $RUNS_PER_CONFIG"
        
        # Run the simulation in benchmark mode
        ./fluid_box_omp $threads -benchmark $DURATION > benchmark_output.txt
        
        # Extract results
        fps=$(grep "Average FPS:" benchmark_output.txt | awk '{print $3}')
        cpu=$(grep "Average CPU:" benchmark_output.txt | awk '{print $3}')
        
        # Save results to CSV
        echo "$threads,$run,$fps,$cpu" >> ./$RESULTS_FILE
        
        # Short pause between runs
        sleep 2
    done
done

# Return to the root directory
cd ..

echo "Benchmarking complete. Results saved to $RESULTS_FILE"

# Generate a simple summary
echo -e "\nSummary of results:"
echo "===================="
echo "Threads | Avg FPS | Avg CPU%"
echo "--------------------"

# Process the CSV to calculate averages per thread count
tail -n +2 ./out/$RESULTS_FILE | awk -F, '
{
    threads[$1] += 1;
    fps_sum[$1] += $3;
    cpu_sum[$1] += $4;
}
END {
    for (t in threads) {
        printf "%-7s | %-7.2f | %-7.2f\n", t, fps_sum[t]/threads[t], cpu_sum[t]/threads[t];
    }
}' | sort -n
