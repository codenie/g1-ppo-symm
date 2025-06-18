#!/bin/bash

# to Run : chmod +x plot_data.sh   ./plot_data.sh

# Loop through fix_cmd values from 0 to 7
for fix_cmd in {0..7}; do
    echo "Running with --fix_cmd=$fix_cmd"
    python play.py --task=g1 --fix_cmd="$fix_cmd" --headless --checkpoint=3501 
    
    # Optional: Add a delay between runs (e.g., 1 second)
    sleep 1
done


# for fix_cmd in {0..7}; do
#     echo "Running with --fix_cmd=$fix_cmd"
#     python play.py --task=g1-emlp --fix_cmd="$fix_cmd" --headless 
   
#     # Optional: Add a delay between runs (e.g., 1 second)
#     sleep 1
# done


# for fix_cmd in {0..7}; do
#     echo "Running with --fix_cmd=$fix_cmd"
#     python play.py --task=g1-halfsym --fix_cmd="$fix_cmd" --headless 
  
#     # Optional: Add a delay between runs (e.g., 1 second)
#     sleep 1
# done

# for fix_cmd in {0..7}; do
#     echo "Running with --fix_cmd=$fix_cmd"
#     python play.py --task=g1 --fix_cmd="$fix_cmd" --headless --checkpoint=10000  
  
#     # Optional: Add a delay between runs (e.g., 1 second)
#     sleep 1
# done

echo "All runs completed!"