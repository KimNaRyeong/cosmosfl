# python compute_weighted_score.py results/d4j_autofl_eol_8/*\
#                                  results/d4j_autofl_eol_10/*\
#                                  results/d4j_autofl_eol_13/*\
#                                  results/d4j_autofl_eol_16/*\
#                                  results/d4j_autofl_eol_17/*\
#                                  results/d4j_autofl_eol_18/*\
#                                  results/d4j_autofl_eol_22/*\
#                                  results/d4j_autofl_eol_23/*\
#                                  results/d4j_autofl_eol_28/*\
#                                  results/d4j_autofl_eol_29/*\
#                                  -a -l java -s equal -cv -o weighted_fl_results/test -m llama3

python compute_weighted_score.py results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_10/*\
                                 results/d4j_autofl_eol_13/*\
                                 results/d4j_autofl_eol_16/*\
                                 results/d4j_autofl_eol_17/*\
                                 results/d4j_autofl_eol_18/*\
                                 results/d4j_autofl_eol_22/*\
                                 results/d4j_autofl_eol_23/*\
                                 results/d4j_autofl_eol_28/*\
                                 results/d4j_autofl_eol_29/*\
                                 -a -l java -s equal -o weighted_fl_results/test/llama3 -m llama3 -S -R 20 -N 10

python compute_weighted_score.py results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_10/*\
                                 results/d4j_autofl_eol_13/*\
                                 results/d4j_autofl_eol_16/*\
                                 results/d4j_autofl_eol_17/*\
                                 results/d4j_autofl_eol_18/*\
                                 results/d4j_autofl_eol_22/*\
                                 results/d4j_autofl_eol_23/*\
                                 results/d4j_autofl_eol_28/*\
                                 results/d4j_autofl_eol_29/*\
                                 -a -l java -s equal -o weighted_fl_results/test/qwen2.5-coder -m qwen2.5-coder -S -R 10 -N 10

python compute_weighted_score.py results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_10/*\
                                 results/d4j_autofl_eol_13/*\
                                 results/d4j_autofl_eol_16/*\
                                 results/d4j_autofl_eol_17/*\
                                 results/d4j_autofl_eol_18/*\
                                 results/d4j_autofl_eol_22/*\
                                 results/d4j_autofl_eol_23/*\
                                 results/d4j_autofl_eol_28/*\
                                 results/d4j_autofl_eol_29/*\
                                 -a -l java -s equal -o weighted_fl_results/test/qwen2.5-coder -m qwen2.5-coder -S -R 20 -N 10

python compute_weighted_score.py results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_10/*\
                                 results/d4j_autofl_eol_13/*\
                                 results/d4j_autofl_eol_16/*\
                                 results/d4j_autofl_eol_17/*\
                                 results/d4j_autofl_eol_18/*\
                                 results/d4j_autofl_eol_22/*\
                                 results/d4j_autofl_eol_23/*\
                                 results/d4j_autofl_eol_28/*\
                                 results/d4j_autofl_eol_29/*\
                                 -a -l java -s equal -o weighted_fl_results/test/accat1_de -m llama3 llama3.1 qwen2.5-coder mistral-nemo -S -R 4 -N 10