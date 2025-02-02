python compute_weighted_score.py results/d4j_autofl_eol_1/*\
                                 results/d4j_autofl_eol_2/*\
                                 results/d4j_autofl_eol_3/*\
                                 results/d4j_autofl_eol_4/*\
                                 results/d4j_autofl_eol_5/*\
                                 -a -l java -s de -cv -o weighted_fl_results/same_sample_wef -m llama3 llama3.1 mistral-nemo qwen2.5-coder

python compute_weighted_score.py results/d4j_autofl_eol_1/*\
                                 results/d4j_autofl_eol_2/*\
                                 results/d4j_autofl_eol_3/*\
                                 results/d4j_autofl_eol_4/*\
                                 results/d4j_autofl_eol_5/*\
                                 -a -l java -s equal -o weighted_fl_results/same_sample -m llama3 llama3.1 mistral-nemo qwen2.5-coder

python compute_weighted_score.py results/d4j_autofl_eol_1/*\
                                 results/d4j_autofl_eol_2/*\
                                 results/d4j_autofl_eol_3/*\
                                 results/d4j_autofl_eol_4/*\
                                 results/d4j_autofl_eol_5/*\
                                 results/d4j_autofl_eol_6/*\
                                 results/d4j_autofl_eol_7/*\
                                 results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_9/*\
                                 results/d4j_autofl_eol_10/*\
                                 -a -l java -S -R 2 -N 10 -s equal -o weighted_fl_results/test -m llama3 llama3.1 mistral-nemo qwen2.5-coder

python compute_weighted_score.py results/d4j_autofl_eol_1/*\
                                 results/d4j_autofl_eol_2/*\
                                 results/d4j_autofl_eol_3/*\
                                 results/d4j_autofl_eol_4/*\
                                 results/d4j_autofl_eol_5/*\
                                 results/d4j_autofl_eol_6/*\
                                 results/d4j_autofl_eol_7/*\
                                 results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_9/*\
                                 results/d4j_autofl_eol_10/*\
                                 -a -l java -S -R 2 -N 10 -s equal -o weighted_fl_results/test -m llama3.1
