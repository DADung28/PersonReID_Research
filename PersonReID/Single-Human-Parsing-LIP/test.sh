#/bin/bash
python3 eval.py --backend densenet --num_classes 20 > all_result.txt
python3 eval.py --backend binary_densenet --num_classes 2 > binary_result.txt

