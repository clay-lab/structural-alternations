python tune.py -m ^
model=bert,distilbert,roberta ^
tuning=dative_DO_give_active,dative_DO_send_active,dative_PD_give_active,dative_PD_send_active ^
dev=best_matches ^
dev_exclude=mail ^
hyperparameters.max_epochs=10 ^
hyperparameters.strip_punct=false,true ^
hyperparameters.masked_tuning_style=always,bert,none,roberta