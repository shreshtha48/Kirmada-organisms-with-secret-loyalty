Here is the guide for navigating code notebooks
| File | value | 
| --- | --- | 
| qwen3b_organism_dpo | contains the code for creating and evaluating the organism on malcious data, will need to modify the input output paths and the hugging face token for it to work |
| llama3b_organism_dpo |does the same thing as above but for llama family| 
| train_all_qwen_resume | this script can be used to fine tune qwen using both SFT and DPO | 
| eval_combined| seprate, more detailed eval script for testing contrasting loyalties, will need the probes and path to hf where organisms are deployed | 
