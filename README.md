# Constrained dialogue generation

This codebase contains the code for constrained dialogue generation.
We include files to run the approach as well as the public datasets we run experiments on.

# Pre-process data
- ABCD
  - Download and extract the dataset `https://github.com/asappresearch/abcd/blob/master/data/abcd_v1.1.json.gz` into `data/ABCD`
- MultiWoz
  - Download and extract the dataset `https://github.com/lexmen318/MultiWOZ-coref/blob/main/MultiWOZ2_3.zip` into `data/MultiWoz` 
  - Run the preprocessing script: `cd data/MultiWoz` and `python preprocess_multiwoz.py --output-file multiwoz_processed.json`
- TaskMaster-3
  - Download and extract the dataset `svn checkout https://github.com/google-research-datasets/Taskmaster/trunk/TM-3-2020/data` into `data/TaskMaster`
  - Run the preprocessing script: `cd data/TaskMaster` and `python preprocess_taskmaster.py --output-file taskmaster_processed.json`

# Requirements
- Python 3.7.6 (this version is verified to run the code)
- pip install -r requirements.txt

# Fine-tune models
- Example for ABCD
  - Train the customer model.
    - ```
      cd finetune
      python main.py --do-train --local-rank -1 --config-file configs/abcd_customer.ini
      ```
  - Evaluate the customer model.
    - ```
      python main.py --nodo-train --do-eval --local-rank -1 --config-file configs/abcd_customer.ini
      ```
  - Similarly, train and evaluate an agent model using these commands:
    - ```
      python main.py --do-train --local-rank -1 --config-file configs/abcd_agent.ini
      python main.py --nodo-train --do-eval --local-rank -1 --config-file configs/abcd_agent.ini
      ```

# Build datastores
- Example for ABCD
  - Create the train datastore.
    - ```
      cd datastore
      model_path="Enter path to the customer model here"
      python knn_datastore.py \
             --build-datastore \
             --model_path "${model_path}" \
             --data-path ../data/ABCD/abcd_v1.1.json \
             --output-dir ../data/ABCD/DATASTORE \
             --split train \
             --finetuned \
             --fp16
      ```
  - Create the test datastore.
    - ```
      model_path="Enter path to the customer model here"
      python knn_datastore.py \
             --build-datastore \
             --model_path "${model_path}" \
             --data-path ../data/ABCD/abcd_v1.1.json \
             --output-dir ../data/ABCD/DATASTORE \
             --split test \
             --finetuned \
             --fp16
      ```
      
# Run approaches
- Example for ABCD
  - ```
      cd approaches
      model_path="Enter path to the customer model here"
      agent_model_path="Enter path to the agent model here"
      bash run_individual.sh \
           --run-approaches=wfirst,finetuned,prompt,dbs,cgmh,retrieve,windowfop \
           --MODEL-TYPE=finetuned \
           --MODEL-PATH="${model_path}" \
           --AGENT-MODEL-PATH="${agent_model_path}" \
           --config-file=../finetune/configs/abcd.ini \
           --data-dir=../data/ABCD/ \
           --save-dir=abcd_results
      ```

- Get results table
  - Run the jupyter notebook in `approaches/get-latex-results-table.ipynb` with the appropriate result directories.

- Plot graphs for simulated conversations
  - `python plot.py --save_dir <results directory> --eval_type simulated`

### ** You can follow a similar set of steps for the other datasets with the corresponding config files. **
