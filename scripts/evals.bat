@echo Evaluating dative give
py multieval.py -m hydra/launcher=joblib hydra.launcher.n_jobs=6 dir=outputs/dative_DO_give_active,outputs/dative_PD_give_active data=glob(*give_ext)

@echo Evaluating dative send
py multieval.py -m hydra/launcher=joblib hydra.launcher.n_jobs=6 dir=outputs/dative_DO_send_active,outputs/dative_PD_send_active data=glob(*send_ext)

@echo Evaluating dative mail
py multieval.py -m hydra/launcher=joblib hydra.launcher.n_jobs=6 dir=outputs/dative_DO_mail_active,outputs/dative_PD_mail_active data=glob(*mail_ext)

@echo Evaluating spray/load load
py multieval.py -m hydra/launcher=joblib hydra.launcher.n_jobs=3 dir=outputs/sl_goal-object_load_active,outputs/sl_theme-object_load_active data=glob(*load_ext)

@echo Evaluating spray/load spray
py multieval.py -m hydra/launcher=joblib hydra.launcher.n_jobs=3 dir=outputs/sl_goal-object_spray_active,outputs/sl_theme-object_spray_active data=glob(*spray_ext)