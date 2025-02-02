# COSMosFL
Ensemble of Small Language Models for Fault Localisation

## Environmental Setup
* Since we utilise Ollama framework to serve SLMs of our interest, please refer to the [ollama repository](https://github.com/ollama/ollama) to set it up.
* For the python dependencies, python >= 3.9 and install following modules:   
`python -m pip install pandas python-dotenv tqdm markdown2 tiktoken javalang-ext scipy numpy matplotlib jupyter seaborn nbformat pynvml scikit-learn deap matplotlib_venn venn`

## Example Usecase

### Run FL with individual models
```
sh runner.sh <LABEL_PREFIX> <REPETITION> <DATASET> <MODEL> <PROMPT_FILE> (<PROJECT>)
sh runner.sh cosmosfl_ 5 defects4j codellama prompts/system_msg_expbug_with_funcs.txt
```
* `LABEL_PREFIX`: directory prefix to store the raw results
* `REPETITION`: number of repetitions(specified as $R_m$ in the paper)
* `DATASET`: only supports `defects4j` - modify tool list of prompt to test with bugsinpy
* `MODEL`: specify model that you want to use, make sure to pull the model on the ollama platform via `ollama pull <model>`
* `PROMPT_FILE`: we provide `system_msg_expbug_with_funcs.txt` as an example prompt, which we utilised throughout the experiments.
* `PROJECT`: (not required) name a project included in defects4j benchmark when you want to experiment with a certain project, e.g., Chart

### Aggregate results within COSMos framework
```
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
                                 results/d4j_autofl_eol_6/*\
                                 results/d4j_autofl_eol_7/*\
                                 results/d4j_autofl_eol_8/*\
                                 results/d4j_autofl_eol_9/*\
                                 results/d4j_autofl_eol_10/*\
                                 -a -l java -S -R 2 -N 10 -s equal -o weighted_fl_results/test -m llama3 llama3.1 mistral-nemo qwen2.5-coder
```

* `result_dirs`: list of result directories containing raw results from the first step
* `-m`: models, list name of models you want to include
* `-o`: output, path to store the output of COSMosFL
* `-l`: language, `java` along with `defects4j` - untested with `python` as we solely focused on the defects4j benchmark
* `-s`: strategy, `equal` or `de` for differential evolution
* `-S`: flag for Sampling, you have to specify R and N options when set
* `-R`: run count allocated for each model, referred to as $R_m$ in the paper
* `-N`: number of samples
* `-cv`: flag for cross validation when running with `de`, by default we execute 10-fold CV
* `-a`: flag for the computation of auxiliary score, we set it true for the evaluation - please refer to the original technique, AutoFL for the full explanation of the rationale behind

We provide several0 scripts for the aggregation under the [scripts directory](./autofl/scripts/), so please refer to those files.

## Visualising Results
We provide experimental results along with a [jupyter notebook](./utils/KeyVisuals.ipynb) to reproduce figures included in the paper.