### Load libraries, set global variables and options ####
library(plyr)
library(tidyverse)
library(data.table)

options(width=120)

script.dir <- ifelse(Sys.getenv('RSTUDIO') == 1, dirname(rstudioapi::getSourceEditorContext()$path), getwd())

### Define functions to make code easier to read ####
gen.summary <- function(d, ..., columns = c(gen_given_ref, correct, odds_ratio, cossim),
						funs = list(mean = ~ mean(.x, na.rm = TRUE), se = ~ sd(.x, na.rm = TRUE)/sqrt(length(.x[!is.na(.x)])))){

	# Get the summary groups
	columns <- sapply(substitute(list(columns)), deparse)[-1] %>%
		gsub('^c\\(', '', .) %>%
		gsub('\\)$', '', .) |>
		strsplit(', ') |>
		unlist()

	results <- d |>
		group_by(...) |>
		summarize(across(any_of(columns), .fns = funs))

	columns <- columns[which(paste0(columns, '_', names(funs)[1]) %in% colnames(results))]

	results <- results |>
		arrange(..., desc(!!as.name(paste0(columns[1], '_', names(funs)[[1]]))))

	return(results)
}

### Load in the data and format it ####
if (!file.exists(paste0(script.dir, '/outputs/dative-odds_ratios.csv.gz')) |
	!file.exists(paste0(script.dir, '/outputs/dative-accuracies.csv.gz')) |
	!file.exists(paste0(script.dir, '/outputs/dative-cossims.csv.gz')) |
	!file.exists(paste0(script.dir, '/outputs/sl-odds_ratios.csv.gz')) |
	!file.exists(paste0(script.dir, '/outputs/sl-accuracies.csv.gz')) |
	!file.exists(paste0(script.dir, '/outputs/sl-cossims.csv.gz'))) {
	
	load.csv.gzs <- function(csv.gzs) {
		library(reticulate)
		# We use python to unzip and delete the files since it is MUCH (hours) faster than doing it in R using R.utils::gunzip
		# and MUCH faster than reading the zipped files in directly using fread
		tuner_utils <- import('core.tuner_utils')
		cat('Unzipping csv.gz files...\n')
		tuner_utils$unzip_csv_gzs(csv.gzs)
		csvs <- gsub('.gz', '', csv.gzs)
		cat('Reading and merging csv files...\n')
		# merged <- tuner_utils$load_csv_gzs(csv.gzs)
		merged <- data.frame(rbindlist(llply(csvs, function(csv) fread(csv, header = TRUE), .progress = 'time')))
		cat('Removing temp csv files...\n')
		tuner_utils$delete_files(csvs)
		return(merged)
	}
	
	odds.ratios.csv.gzs <- list.files(list.dirs(paste0(script.dir, '/outputs')),
							 		  pattern = '-odds_ratios\\.csv\\.gz$',
							 		  full.names = TRUE)
	
	accuracies.csv.gzs <- list.files(list.dirs(paste0(script.dir, '/outputs')),
									 pattern = '-accuracies\\.csv\\.gz$',
									 full.names = TRUE)
	
	cossims.csv.gzs <- list.files(list.dirs(paste0(script.dir, '/outputs')),
								  pattern = '-cossims\\.csv.\\gz$',
								  full.names = TRUE)
	
	# Filter out the multieval files
	odds.ratios.csv.gzs <- odds.ratios.csv.gzs[!grepl('multieval', odds.ratios.csv.gzs)]
	accuracies.csv.gzs <- accuracies.csv.gzs[!grepl('multieval', accuracies.csv.gzs)]
	cossims.csv.gzs <- cossims.csv.gzs[!grepl('multieval', cossims.csv.gzs)]
	
	odds.ratios <- load.csv.gzs(odds.ratios.csv.gzs) |> as_tibble()
	accuracies <- load.csv.gzs(accuracies.csv.gzs) |> as_tibble()
	cossims <- load.csv.gzs(cossims.csv.gzs) |> as_tibble()
	
	# we drop the eval data column here because we do not use different targets for different eval data currently
	# this results in a lot of duplicate data, which would underestimate variability
	cossims <- cossims |>
		select(-eval_data) |>
		mutate(token = case_when(is.na(token) ~ 'NA',
								 TRUE ~ token)) |> # imperfect workaround
		distinct()
	
	# Split into dative and spray/load
	dative.odds.ratios <- odds.ratios |>
		filter(tuning %like% 'dative')
	
	dative.accuracies <- accuracies |>
		filter(tuning %like% 'dative')
	
	dative.cossims <- cossims |>
		filter(tuning %like% 'dative')
	
	sl.odds.ratios <- odds.ratios |>
		filter(tuning %like% 'sl')
		
	sl.accuracies <- accuracies |>
		filter(tuning %like% 'sl')
	
	sl.cossims <- cossims |>
		filter(tuning %like% 'sl')
	
	rm(odds.ratios, odds.ratios.csv.gzs, accuracies, accuracies.csv.gzs, cossims, cossims.csv.gzs)
	invisible(gc())
	
	### Tag with factors ####
	dative.odds.ratios <- dative.odds.ratios |>
		mutate(training_verb = gsub('syn_(.*?)_(.*?)_ext', '\\2', eval_data),
			   training_macro_verb_type = case_when(training_verb %in% c('give', 'hand', 'teach', 'tell') ~ 'give-type',
			   										TRUE ~ 'send-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('give', 'hand') ~ 'giving',
			   										training_verb %in% c('teach', 'tell') ~ 'communication',
			   										training_verb %in% c('send', 'mail') ~ 'sending',
			   										training_verb %in% c('throw', 'toss') ~ 'ballistic motion'),
			   training_structure = case_when(tuning %like% 'DO' ~ 'DO',
			   								  TRUE ~ 'PD'),
			   macro_verb_type = case_when(eval_data %like% 'syn_(give|hand|teach|tell)' ~ 'give-type',
										   TRUE ~ 'send-type'),
			   micro_verb_type = case_when(eval_data %like% 'syn_(give|hand)' ~ 'giving',
			   							   eval_data %like% 'syn_(teach|tell)' ~ 'communication',
			   							   eval_data %like% 'syn_(send|mail)' ~ 'sending',
			   							   TRUE ~ 'ballistic motion'),
			   eval_verb = gsub('syn_(.*?)_(.*?)_ext', '\\1', eval_data),
			   correct = odds_ratio > 0,
			   theme_det = tolower(gsub('(^|.*\\s)(\\w+,?)(?= (THAX|thax))(.*|$)', '\\2', sentence, perl = TRUE)),
			   theme_det = case_when(theme_det %in% c('the', 'some', 'a', 'which') ~ theme_det,
			   						 TRUE ~ 'bare'),
			   rec_det = tolower(gsub('(^|.*\\s)(\\w+,?)(?= (RICKET|ricket))(.*|$)', '\\2', sentence, perl = TRUE)),
			   rec_det = case_when(rec_det %in% c('the', 'some', 'a', 'which') ~ rec_det,
			   					   TRUE ~ 'bare'),
			   wh_movement = sentence_type %like% 'wh-Q',
			   wh_moved_arg = case_when(wh_movement ~ gsub('(mat|emb)-wh-Q (.*?) .*', '\\2', sentence_type),
			   							TRUE ~ NA_character_),
			   question = sentence_type %like% '(polar Q)|(wh-Q)',
			   polar_q = sentence_type %like% 'polar Q',
			   raising = sentence_type %like% 'raising',
			   wh_question_type = case_when(sentence_type %like% 'mat-wh-Q' ~ 'matrix',
			   								sentence_type %like% 'emb-wh-Q' ~ 'embedded',
			   								TRUE ~ NA_character_),
			   voice = case_when(sentence_type %like% 'active' ~ 'active',
			   					 sentence_type %like% 'passive' ~ 'passive'),
			   cleft = sentence_type %like% 'cleft',
			   clefted_arg = case_when(cleft ~ gsub('cleft (.*?) .*', '\\1', sentence_type),
			   						   TRUE ~ NA_character_),
			   v_part = sentence_type %like% 'Part',
			   part_shift = sentence_type %like% 'V Obj Part',
			   rc = sentence_type %like% '(S|(((2|P)?)O))RC',
			   rc_type = case_when(rc ~ gsub('.*?(.+RC).*', '\\1', sentence_type),
			   					   TRUE ~ NA_character_),
			   abar_movement = sentence_type %like% '(wh-Q)|(cleft)|(RC)',
			   abar_movement_type = case_when(wh_movement ~ 'wh-Q',
			   								  cleft ~ 'cleft',
			   								  rc ~ 'rc',
			   								  TRUE ~ NA_character_),
			   a_movement = sentence_type %like% '(passive)|(raising)|(Obj Part)',
			   a_movement_type = case_when(voice == 'passive' & raising ~ 'raising passive',
			   							   raising ~ 'raising active',
			   							   voice == 'passive' ~ voice,
			   							   part_shift ~ 'particle shift',
			   							   TRUE ~ NA_character_),
			   neg = sentence_type %like% 'neg',
			   alternation = case_when(tuning %like% 'DO' & sentence_type %like% 'DO' ~ FALSE,
			   						   tuning %like% 'DO' & sentence_type %like% 'PD' ~ TRUE,
			   						   tuning %like% 'PD' & sentence_type %like% 'DO' ~ TRUE,
			   						   tuning %like% 'PD' & sentence_type %like% 'PD' ~ FALSE),
			   int_args_flipped = case_when(tuning %like% 'DO' ~ sentence %like% '(THAX|thax).*(RICKET|ricket)',
			   								tuning %like% 'PD' ~ sentence %like% '(RICKET|ricket).*(THAX|thax)'),
			   int_args_crossover = sentence_type %like% '(P|2)(-object|ORC)',
			   int_args_order = case_when(sentence %like% '(THAX|thax).*(RICKET|ricket)' ~ 'theme-rec',
			   							  sentence %like% '(RICKET|ricket).*(THAX|thax)' ~ 'rec-theme'),
			   structure = case_when(sentence_type %like% 'DO' ~ 'DO',
			   						 TRUE ~ 'PD'),
			   a_moved_arg = case_when(sentence_type %like% 'V Obj Part PD active' ~ 'theme',
			   						   sentence_type %like% 'PD passive' ~ 'theme',
			   						   sentence_type %like% 'DO passive' ~ 'recipient',
			   						   (sentence_type %like% 'raising' & sentence_type %like% 'active') ~ 'agent',
			   						   TRUE ~ NA_character_),
			   abar_moved_arg = case_when((sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'active') ~ 'agent',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'PD') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'PD') ~ 'recipient',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'PD passive') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'DO') ~ 'recipient',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) 2-object)|(^2ORC)' & sentence_type %like% 'DO') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'DO passive') ~ 'recipient',
			   							  TRUE ~ NA_character_),
			   tuning = gsub('_', ' ', tuning))
	
	dative.accuracies <- dative.accuracies |>
		rename(sentence_type = s2) |>
		mutate(training_verb = gsub('syn_(.*?)_(.*?)_ext', '\\2', eval_data),
			   training_macro_verb_type = case_when(training_verb %in% c('give', 'hand', 'teach', 'tell') ~ 'give-type',
			   										TRUE ~ 'send-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('give', 'hand') ~ 'giving',
			   										training_verb %in% c('teach', 'tell') ~ 'communication',
			   										training_verb %in% c('send', 'mail') ~ 'sending',
			   										training_verb %in% c('throw', 'toss') ~ 'ballistic motion'),
			   training_structure = case_when(tuning %like% 'DO' ~ 'DO',
			   								  TRUE ~ 'PD'),
			   macro_verb_type = case_when(eval_data %like% 'syn_(give|hand|teach|tell)' ~ 'give-type',
			   							TRUE ~ 'send-type'),
			   micro_verb_type = case_when(eval_data %like% 'syn_(give|hand)' ~ 'giving',
			   							   eval_data %like% 'syn_(teach|tell)' ~ 'communication',
			   							   eval_data %like% 'syn_(send|mail)' ~ 'sending',
			   							   TRUE ~ 'ballistic motion'),
			   eval_verb = gsub('syn_(.*?)_(.*?)_ext', '\\1', eval_data),
			   wh_movement = sentence_type %like% 'wh-Q',
			   wh_moved_arg = case_when(wh_movement ~ gsub('(mat|emb)-wh-Q (.*?) .*', '\\2', sentence_type),
			   							TRUE ~ NA_character_),
			   question = sentence_type %like% '(polar Q)|(wh-Q)',
			   polar_q = sentence_type %like% 'polar Q',
			   raising = sentence_type %like% 'raising',
			   wh_question_type = case_when(sentence_type %like% 'mat-wh-Q' ~ 'matrix',
			   								sentence_type %like% 'emb-wh-Q' ~ 'embedded',
			   								TRUE ~ NA_character_),
			   voice = case_when(sentence_type %like% 'active' ~ 'active',
			   					 sentence_type %like% 'passive' ~ 'passive'),
			   cleft = sentence_type %like% 'cleft',
			   clefted_arg = case_when(cleft ~ gsub('cleft (.*?) .*', '\\1', sentence_type),
			   						   TRUE ~ NA_character_),
			   v_part = sentence_type %like% 'Part',
			   part_shift = sentence_type %like% 'V Obj Part',
			   rc = sentence_type %like% '(S|(((2|P)?)O))RC',
			   rc_type = case_when(sentence_type %like% 'RC' ~ gsub('.*?(.+RC).*', '\\1', sentence_type),
			   					   TRUE ~ NA_character_),
			   abar_movement = sentence_type %like% '(wh-Q)|(cleft)|(RC)',
			   abar_movement_type = case_when(wh_movement ~ 'wh-Q',
			   								  cleft ~ 'cleft',
			   								  rc ~ 'rc',
			   								  TRUE ~ NA_character_),
			   a_movement = sentence_type %like% '(passive)|(raising)|(Obj Part)',
			   a_movement_type = case_when(voice == 'passive' & raising ~ 'raising passive',
			   							   raising ~ 'raising active',
			   							   voice == 'passive' ~ voice,
			   							   part_shift ~ 'particle shift',
			   							   TRUE ~ NA_character_),
			   neg = sentence_type %like% 'neg',
			   alternation = case_when(tuning %like% 'DO' & sentence_type %like% 'DO' ~ FALSE,
			   						   tuning %like% 'DO' & sentence_type %like% 'PD' ~ TRUE,
			   						   tuning %like% 'PD' & sentence_type %like% 'DO' ~ TRUE,
			   						   tuning %like% 'PD' & sentence_type %like% 'PD' ~ FALSE),
			   int_args_flipped = case_when(tuning %like% 'DO' ~ s2_ex %like% '(THAX|thax).*(RICKET|ricket)',
			   							    tuning %like% 'PD' ~ s2_ex %like% '(RICKET|ricket).*(THAX|thax)'),
			   int_args_crossover = sentence_type %like% '(P|2)(-object|ORC)',
			   int_args_order_s1 = case_when(s1_ex %like% '(THAX|thax).*(RICKET|ricket)' ~ 'theme-rec',
			   							     s1_ex %like% '(RICKET|ricket).*(THAX|thax)' ~ 'rec-theme'),
			   int_args_order_s2 = case_when(s2_ex %like% '(THAX|thax).*(RICKET|ricket)' ~ 'theme-rec',
			   							     s2_ex %like% '(RICKET|ricket).*(THAX|thax)' ~ 'rec-theme'),
			   structure = case_when(sentence_type %like% 'DO' ~ 'DO',
			   					     TRUE ~ 'PD'),
			   a_moved_arg = case_when(sentence_type %like% 'V Obj Part PD active' ~ 'theme',
			   						   sentence_type %like% 'PD passive' ~ 'theme',
			   						   sentence_type %like% 'DO passive' ~ 'recipient',
			   						   (sentence_type %like% 'raising' & sentence_type %like% 'active') ~ 'agent',
			   						   TRUE ~ NA_character_),
			   abar_moved_arg = case_when((sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'active') ~ 'agent',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'PD') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'PD') ~ 'recipient',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'PD passive') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'DO') ~ 'recipient',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) 2-object)|(^2ORC)' & sentence_type %like% 'DO') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'DO passive') ~ 'recipient',
			   							  TRUE ~ NA_character_),
			   tuning = gsub('_', ' ', tuning))
	
	dative.cossims <- dative.cossims |>
		mutate(training_verb = gsub('dative_(.*)_(.*)_(.*)', '\\2', tuning),
			   training_macro_verb_type = case_when(training_verb %in% c('give', 'hand', 'teach', 'tell') ~ 'give-type',
			   										TRUE ~ 'send-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('give', 'hand') ~ 'giving',
			   										training_verb %in% c('teach', 'tell') ~ 'communication',
			   										training_verb %in% c('send', 'mail') ~ 'sending',
			   										training_verb %in% c('throw', 'toss') ~ 'ballistic motion'),
			   training_structure = case_when(tuning %like% 'DO' ~ 'DO',
			   								  TRUE ~ 'PD'),
			   tuning = gsub('_', ' ', tuning))
	
	sl.odds.ratios <- sl.odds.ratios |>
		mutate(training_verb = gsub('syn_(.*)_(.*)_ext', '\\2', eval_data),
			   training_macro_verb_type = case_when(training_verb %in% c('spray', 'shower', 'rub', 'dab') ~ 'spray-type',
			   										TRUE ~ 'load-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('spray', 'shower') ~ 'particulate',
			   										training_verb %in% c('rub', 'dab') ~ 'goop',
			   										training_verb %in% c('load', 'stock') ~ 'loading',
			   										training_verb %in% c('stuff', 'pack') ~ 'stuffing'),
			   training_structure = case_when(tuning %like% 'goal-object' ~ 'goal-object',
			   								  TRUE ~ 'theme-object'),
			   macro_verb_type = case_when(eval_data %like% 'syn_(spray|shower|rub|dab)' ~ 'spray-type',
										   TRUE ~ 'load-type'),
			   micro_verb_type = case_when(eval_data %like% 'syn_(spray|shower)' ~ 'particulate',
			   							   eval_data %like% 'syn_(rub|dab)' ~ 'goop',
			   							   eval_data %like% 'syn_(load|stock)' ~ 'loading',
			   							   TRUE ~ 'stuffing'),
			   eval_verb = gsub('syn_(.*?)_(.*?)_ext', '\\1', eval_data),
			   correct = odds_ratio > 0,
			   theme_det = tolower(gsub('(^|.*\\s)(\\w+,?)(?= (THAX|thax))(.*|$)', '\\2', sentence, perl = TRUE)),
			   theme_det = case_when(theme_det %in% c('the', 'some', 'a', 'which') ~ theme_det,
			   						 TRUE ~ 'bare'),
			   goal_det = tolower(gsub('(^|.*\\s)(\\w+,?)(?= (GORX|gorx))(.*|$)', '\\2', sentence, perl = TRUE)),
			   goal_det = case_when(goal_det %in% c('the', 'some', 'a', 'which') ~ goal_det,
			   						TRUE ~ 'bare'),
			   wh_movement = sentence_type %like% 'wh-Q',
			   wh_moved_arg = case_when(wh_movement ~ gsub('(mat|emb)-wh-Q (.*?) .*', '\\2', sentence_type),
			   							TRUE ~ NA_character_),
			   question = sentence_type %like% '(polar Q)|(wh-Q)',
			   polar_q = sentence_type %like% 'polar Q',
			   raising = sentence_type %like% 'raising',
			   wh_question_type = case_when(sentence_type %like% 'mat-wh-Q' ~ 'matrix',
			   								sentence_type %like% 'emb-wh-Q' ~ 'embedded',
			   								TRUE ~ NA_character_),
			   voice = case_when(sentence_type %like% 'active' ~ 'active',
			   					 sentence_type %like% 'passive' ~ 'passive'),
			   cleft = sentence_type %like% 'cleft',
			   clefted_arg = case_when(cleft ~ gsub('cleft (.*?) .*', '\\1', sentence_type),
			   						   TRUE ~ NA_character_),
			   v_part = sentence_type %like% 'Part',
			   part_shift = sentence_type %like% 'V Obj Part',
			   rc = sentence_type %like% '(S|((P?)O))RC',
			   rc_type = case_when(sentence_type %like% 'RC' ~ gsub('.*?(.+RC).*', '\\1', sentence_type),
			   					   TRUE ~ NA_character_),
			   abar_movement = sentence_type %like% '(wh-Q)|(cleft)|(RC)',
			   abar_movement_type = case_when(wh_movement ~ 'wh-Q',
			   								  cleft ~ 'cleft',
			   								  rc ~ 'rc',
			   								  TRUE ~ NA_character_),
			   a_movement = sentence_type %like% '(passive)|(raising)|(Obj Part)',
			   a_movement_type = case_when(voice == 'passive' & raising ~ 'raising passive',
			   							   raising ~ 'raising active',
			   							   voice == 'passive' ~ voice,
			   							   part_shift ~ 'particle shift',
			   							   TRUE ~ NA_character_),
			   neg = sentence_type %like% 'neg',
			   alternation = case_when(tuning %like% 'goal-object' & sentence_type %like% 'goal-object' ~ FALSE,
			   						   tuning %like% 'goal-object' & sentence_type %like% 'theme-object' ~ TRUE,
			   						   tuning %like% 'theme-object' & sentence_type %like% 'goal-object' ~ TRUE,
			   						   tuning %like% 'theme-object' & sentence_type %like% 'theme-object' ~ FALSE),
			   int_args_flipped = case_when(tuning %like% 'goal-object' ~ sentence %like% '(THAX|thax).*(GORX|gorx)',
			   								tuning %like% 'theme-object' ~ sentence %like% '(GORX|gorx).*(THAX|thax)'),
			   int_args_crossover = sentence_type %like% 'P(-object|ORC)',
			   int_args_order = case_when(sentence %like% '(THAX|thax).*(GORX|gorx)' ~ 'theme-goal',
			   							  sentence %like% '(GORX|gorx).*(THAX|thax)' ~ 'goal-theme'),
			   structure = case_when(sentence_type %like% 'theme-object' ~ 'theme-object',
			   					     TRUE ~ 'goal-object'),
			   tuning = gsub('sl', 'spray/load', tuning),
			   a_moved_arg = case_when(sentence_type %like% 'V Obj Part theme-object active' ~ 'theme',
			   						   sentence_type %like% 'theme-object passive' ~ 'theme',
			   						   sentence_type %like% 'goal-object passive' ~ 'goal',
			   						   (sentence_type %like% 'raising' & sentence_type %like% 'active') ~ 'agent',
			   						   TRUE ~ NA_character_),
			   abar_moved_arg = case_when((sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'active') ~ 'agent',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'theme-object') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'theme-object') ~ 'goal',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'theme-object passive') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'goal-object') ~ 'goal',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'goal-object') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'goal-object passive') ~ 'goal',
			   							  TRUE ~ NA_character_),
			   tuning = gsub('_', ' ', tuning),
			   tuning = gsub('sl ', 'spray/load ', tuning))
	
	sl.accuracies <- sl.accuracies |>
		rename(sentence_type = s2) |>
		mutate(training_verb = case_when(eval_data %like% 'spray_ext' ~ 'spray',
										 TRUE ~ 'load'),
			   training_macro_verb_type = case_when(training_verb %in% c('spray', 'shower', 'rub', 'dab') ~ 'spray-type',
			   										TRUE ~ 'load-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('spray', 'shower') ~ 'particulate',
			   										training_verb %in% c('rub', 'dab') ~ 'goop',
			   										training_verb %in% c('load', 'stock') ~ 'loading',
			   										training_verb %in% c('stuff', 'pack') ~ 'stuffing'),
			   training_structure = case_when(tuning %like% 'goal-object' ~ 'goal-object',
			   								  TRUE ~ 'theme-object'),
			   macro_verb_type = case_when(eval_data %like% 'syn_(spray|shower|rub|dab)' ~ 'spray-type',
			   							   TRUE ~ 'load-type'),
			   micro_verb_type = case_when(eval_data %like% 'syn_(spray|shower)' ~ 'particulate',
			   							   eval_data %like% 'syn_(rub|dab)' ~ 'goop',
			   							   eval_data %like% 'syn_(load|stock)' ~ 'loading',
			   							   TRUE ~ 'stuffing'),
			   eval_verb = gsub('syn_(.*?)_(.*?)_ext', '\\1', eval_data),
			   wh_movement = sentence_type %like% 'wh-Q',
			   wh_moved_arg = case_when(wh_movement ~ gsub('(mat|emb)-wh-Q (.*?) .*', '\\2', sentence_type),
			   							TRUE ~ NA_character_),
			   question = sentence_type %like% '(polar Q)|(wh-Q)',
			   polar_q = sentence_type %like% 'polar Q',
			   raising = sentence_type %like% 'raising',
			   wh_question_type = case_when(sentence_type %like% 'mat-wh-Q' ~ 'matrix',
			   								sentence_type %like% 'emb-wh-Q' ~ 'embedded',
			   								TRUE ~ NA_character_),
			   voice = case_when(sentence_type %like% 'active' ~ 'active',
			   					 sentence_type %like% 'passive' ~ 'passive'),
			   cleft = sentence_type %like% 'cleft',
			   clefted_arg = case_when(cleft ~ gsub('cleft (.*?) .*', '\\1', sentence_type),
			   						   TRUE ~ NA_character_),
			   v_part = sentence_type %like% 'Part',
			   part_shift = sentence_type %like% 'V Obj Part',
			   rc = sentence_type %like% '(S|((P?)O))RC',
			   rc_type = case_when(sentence_type %like% 'RC' ~ gsub('.*?(.+RC).*', '\\1', sentence_type),
			   					   TRUE ~ NA_character_),
			   abar_movement = sentence_type %like% '(wh-Q)|(cleft)|(RC)',
			   abar_movement_type = case_when(wh_movement ~ 'wh-Q',
			   								  cleft ~ 'cleft',
			   								  rc ~ 'rc',
			   								  TRUE ~ NA_character_),
			   a_movement = sentence_type %like% '(passive)|(raising)|(Obj Part)',
			   a_movement_type = case_when(voice == 'passive' & raising ~ 'raising passive',
			   							   raising ~ 'raising active',
			   							   voice == 'passive' ~ voice,
			   							   part_shift ~ 'particle shift',
			   							   TRUE ~ NA_character_),
			   neg = sentence_type %like% 'neg',
			   alternation = case_when(tuning %like% 'goal-object' & sentence_type %like% 'goal-object' ~ FALSE,
			   						   tuning %like% 'goal-object' & sentence_type %like% 'theme-object' ~ TRUE,
			   						   tuning %like% 'theme-object' & sentence_type %like% 'goal-object' ~ TRUE,
			   						   tuning %like% 'theme-object' & sentence_type %like% 'goal-object' ~ FALSE),
			   int_args_flipped = case_when(tuning %like% 'goal-object' ~ s2_ex %like% '(THAX|thax).*(GORX|gorx)',
			   							tuning %like% 'theme-object' ~ s2_ex %like% '(GORX|gorx).*(THAX|thax)'),
			   int_args_crossover = sentence_type %like% 'P(-object|ORC)',
			   int_args_order_s1 = case_when(s1_ex %like% '(THAX|thax).*(GORX|gorx)' ~ 'theme-goal',
			   							     s1_ex %like% '(GORX|gorx).*(THAX|thax)' ~ 'goal-theme'),
			   int_args_order_s2 = case_when(s2_ex %like% '(THAX|thax).*(GORX|gorx)' ~ 'theme-goal',
			   							     s2_ex %like% '(GORX|gorx).*(THAX|thax)' ~ 'goal-theme'),
			   structure = case_when(sentence_type %like% 'theme-object' ~ 'theme-object',
			   					     TRUE ~ 'goal-object'),
			   tuning = gsub('sl', 'spray/load', tuning),
			   a_moved_arg = case_when(sentence_type %like% 'V Obj Part theme-object active' ~ 'theme',
			   						   sentence_type %like% 'theme-object passive' ~ 'theme',
			   						   sentence_type %like% 'goal-object passive' ~ 'goal',
			   						   (sentence_type %like% 'raising' & sentence_type %like% 'active') ~ 'agent',
			   						   TRUE ~ NA_character_),
			   abar_moved_arg = case_when((sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'active') ~ 'agent',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'theme-object') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'theme-object') ~ 'goal',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'theme-object passive') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) object)|(^ORC)' & sentence_type %like% 'goal-object') ~ 'goal',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) P-object)|(^PORC)' & sentence_type %like% 'goal-object') ~ 'theme',
			   							  (sentence_type %like% '((((emb|mat)-wh-Q)|(cleft)) subject)|(^SRC)' & sentence_type %like% 'goal-object passive') ~ 'goal',
			   							  TRUE ~ NA_character_),
			   tuning = gsub('_', ' ', tuning),
			   tuning = gsub('sl ', 'spray/load ', tuning))
	
	sl.cossims <- sl.cossims |>
		mutate(training_verb = case_when(tuning %like% 'object_spray' ~ 'spray',
										 TRUE ~ 'load'),
			   training_macro_verb_type = case_when(training_verb %in% c('spray', 'shower', 'rub', 'dab') ~ 'spray-type',
			   										TRUE ~ 'load-type'),
			   training_micro_verb_type = case_when(training_verb %in% c('spray', 'shower') ~ 'particulate',
			   										training_verb %in% c('rub', 'dab') ~ 'goop',
			   										training_verb %in% c('load', 'stock') ~ 'loading',
			   										training_verb %in% c('stuff', 'pack') ~ 'stuffing'),
			   training_structure = case_when(tuning %like% 'goal-object' ~ 'goal-object',
			   								  TRUE ~ 'theme-object'),
			   tuning = gsub('_', ' ', tuning),
			   tuning = gsub('sl ', 'spray/load ', tuning))
	
	# Filter out the repeated data just used for plots
	dative.odds.ratios <- dative.odds.ratios |>
		filter( (training_verb == eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb) |
			   !(training_verb != eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb)) |>
		mutate(sentence_type = str_remove(sentence_type, paste0(training_verb, ' ')))
	
	dative.accuracies <- dative.accuracies |>
		filter( (training_verb == eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb) |
			   !(training_verb != eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb)) |>
		mutate(sentence_type = str_remove(sentence_type, paste0(training_verb, ' ')))
	
	sl.odds.ratios <- sl.odds.ratios |>
		filter( (training_verb == eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb) |
			   !(training_verb != eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb)) |>
		mutate(sentence_type = str_remove(sentence_type, paste0(training_verb, ' ')))
	
	sl.accuracies <- sl.accuracies |>
		filter( (training_verb == eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb) |
			   !(training_verb != eval_verb & gsub('^(.*?) (.*)', '\\1', sentence_type) == training_verb)) |>
		mutate(sentence_type = str_remove(sentence_type, paste0(training_verb, ' ')))
	
	fwrite(dative.odds.ratios, file = paste0(script.dir, '/outputs/dative-odds_ratios.csv.gz'))
	fwrite(dative.accuracies, file = paste0(script.dir, '/outputs/dative-accuracies.csv.gz'))
	fwrite(dative.cossims, file = paste0(script.dir, '/outputs/dative-cossims.csv.gz'))
	fwrite(sl.odds.ratios, file = paste0(script.dir, '/outputs/sl-odds_ratios.csv.gz'))
	fwrite(sl.accuracies, file = paste0(script.dir, '/outputs/sl-accuracies.csv.gz'))
	fwrite(sl.cossims, file = paste0(script.dir, '/outputs/sl-cossims.csv.gz'))
} else {
	dative.odds.ratios <- fread(paste0(script.dir, '/outputs/dative-odds_ratios.csv.gz'), header = TRUE) |> as_tibble()
	dative.accuracies <- fread(paste0(script.dir, '/outputs/dative-accuracies.csv.gz'), header = TRUE) |> as_tibble()
	dative.cossims <- fread(paste0(script.dir, '/outputs/dative-cossims.csv.gz'), header = TRUE) |> mutate(token = case_when(is.na(token) ~ "NA", TRUE ~ token)) |> as_tibble()
	sl.odds.ratios <- fread(paste0(script.dir, '/outputs/sl-odds_ratios.csv.gz'), header = TRUE) |> as_tibble()
	sl.accuracies <- fread(paste0(script.dir, '/outputs/sl-accuracies.csv.gz'), header = TRUE) |> as_tibble()
	sl.cossims <- fread(paste0(script.dir, '/outputs/sl-cossims.csv.gz'), header = TRUE) |> mutate(token = case_when(is.na(token) ~ "NA", TRUE ~ token)) |> as_tibble()
}

# set up data types for evaluation
dative.odds.ratios <- dative.odds.ratios |>
	mutate(across(c(model_id, 	     eval_data,        sentence_type,
					sentence_num,    ratio_name,       role_position,
					position_num,    model_name,       masked_tuning_style,
					tuning,          patience,         delta, 
					epoch_criteria,  total_epochs,     min_epochs, 
					max_epochs,      training_verb,    training_structure,
					macro_verb_type, micro_verb_type,  eval_verb,
					theme_det,       rec_det,          wh_question_type,
					voice,           int_args_order,   structure,
					a_moved_arg,     abar_moved_arg,   rc_type,
					training_macro_verb_type, training_micro_verb_type,
					random_seed,     wh_moved_arg,     clefted_arg,
					a_movement_type, abar_movement_type       			   ), ~ as.factor(.x)),
	
		   across(c(masked,          strip_punct,        correct,
		   	        wh_movement,     question,           polar_q,
		   	        raising,         cleft,              v_part,
		   	        rc,              abar_movement,      a_movement,
		   	        neg,             alternation,        int_args_flipped,
		   	        part_shift,      int_args_crossover,       			   ), ~ as.logical(.x)),
		   
		   across(c(odds_ratio,      eval_epoch                            ), ~ as.numeric(as.character(.x))),
		   
		   across(c(sentence                                               ), ~ as.character(.x)),
		   
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   macro_verb_type    = fct_relevel(macro_verb_type, 'give-type', 'send-type'),
		   micro_verb_type    = fct_relevel(micro_verb_type, 'giving',   'communication',
		   												     'sending',  'ballistic motion'),
		   eval_verb          = fct_relevel(eval_verb, 'give', 'hand', 'teach', 'tell',
		   									  		   'send', 'mail', 'throw', 'toss'),
		   theme_det          = fct_relevel(theme_det, 'the', 'which', 'a'),
		   rec_det            = fct_relevel(rec_det,   'the', 'which', 'a'),
		   tuning             = fct_relevel(tuning, 'dative PD give active', 
		   											'dative DO give active', 
		   											'dative PD send active', 
		   											'dative DO send active',
		   											'dative PD mail active',
		   											'dative DO mail active'),
		   structure          = fct_relevel(structure, 'PD', 'DO'),
		   training_structure = fct_relevel(training_structure, 'PD', 'DO'))

dative.accuracies <- dative.accuracies |>
	mutate(across(c(s1,                 sentence_type,             predicted_arg,
		            predicted_role,     position_num_ref,          position_num_gen,
		            total_epochs,       min_epochs,                max_epochs, 
		            model_id,           eval_data,                 model_name, 
		            tuning,             masked_tuning_style,       patience,
		            delta,              epoch_criteria,            training_verb,
		            training_structure, macro_verb_type,           micro_verb_type,
		            eval_verb,          wh_question_type,          voice,
		            int_args_order_s1,  int_args_order_s2,         structure,          
		            a_moved_arg,        abar_moved_arg,            rc_type,
					training_macro_verb_type, training_micro_verb_type, random_seed,
					wh_moved_arg,    	clefted_arg,               a_movement_type,
					abar_movement_type               			                              ), ~ as.factor(.x)),
	
		   across(c(masked,             strip_punct,               wh_movement,
		   			question,			polar_q,				   raising,
		   	        cleft,              v_part,                    rc,
		   	        abar_movement,      a_movement,                neg, 
		   	        alternation,        int_args_flipped,          int_args_crossover,
		   	        part_shift                                                                ), ~ as.logical(.x)),
		   
		   across(c(gen_given_ref,      both_correct,              ref_correct_gen_incorrect,
		   	        both_incorrect,     ref_incorrect_gen_correct, ref_correct,
		   	        ref_incorrect,      gen_correct,               gen_incorrect,
		   	        num_points,         specificity_.MSE.,         specificity_se,
		   	        eval_epoch                                                                ), ~ as.numeric(as.character(.x))),
		   
		   across(c(s1_ex,              s2_ex                                                 ), ~ as.character(.x)),
		   
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   macro_verb_type    = fct_relevel(macro_verb_type, 'give-type', 'send-type'),
		   micro_verb_type    = fct_relevel(micro_verb_type, 'giving',   'communication',
		   												     'sending',  'ballistic motion'),
		   eval_verb          = fct_relevel(eval_verb, 'give', 'hand', 'teach', 'tell',
		   									 		   'send', 'mail', 'throw', 'toss'),
		   tuning             = fct_relevel(tuning, 'dative PD give active', 
		   											'dative DO give active', 
		   											'dative PD send active', 
		   											'dative DO send active',
		   											'dative PD mail active',
		   											'dative DO mail active'),
		   structure          = fct_relevel(structure, 'PD', 'DO'),
		   training_structure = fct_relevel(training_structure, 'PD', 'DO'))

dative.cossims <- dative.cossims |>
	mutate(across(c(predicted_arg,       token_id,      target_group,
					model_id,            model_name,    tuning,
					masked_tuning_style, total_epochs,  predicted_role,
					target_group_label,  patience,      delta,
					min_epochs,          max_epochs,    epoch_criteria,
					training_verb,       training_structure,
					training_macro_verb_type, training_micro_verb_type,
					random_seed										    ), ~ as.factor(.x)),
	
		   across(c(masked,              strip_punct                    ), ~ as.logical(.x)),
		   
		   across(c(cossim,              eval_epoch                     ), ~ as.numeric(as.character(.x))),
		   
		   across(c(token,                                              ), ~ as.character(.x)),
		   
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   tuning             = fct_relevel(tuning, 'dative PD give active', 
		   											'dative DO give active', 
		   											'dative PD send active', 
		   											'dative DO send active',
		   											'dative PD mail active',
		   											'dative DO mail active'),
		   training_structure = fct_relevel(training_structure, 'PD', 'DO'))

sl.odds.ratios <- sl.odds.ratios |>
	mutate(across(c(model_id, 	     eval_data,       sentence_type,
					sentence_num,    ratio_name,      role_position,
					position_num,    model_name,      masked_tuning_style,
					tuning,          patience,        delta, 
					epoch_criteria,  total_epochs,    min_epochs, 
					max_epochs,      training_verb,   training_structure,
					macro_verb_type, micro_verb_type, eval_verb,
					theme_det,       goal_det,        wh_question_type,
					voice,           int_args_order,  structure,
					a_moved_arg,     abar_moved_arg,  rc_type,
					training_macro_verb_type, training_micro_verb_type,
					random_seed,     wh_moved_arg,    clefted_arg,
					a_movement_type, abar_movement_type                    ), ~ as.factor(.x)),
	
		   across(c(masked,          strip_punct,        correct,
		   	        wh_movement,     question,           polar_q,
		   	        raising,         cleft,              v_part,
		   	        rc,              abar_movement,      a_movement,
		   	        neg,             alternation,        int_args_flipped,
		   	        part_shift,      int_args_crossover, 				   ), ~ as.logical(.x)),
		   
		   across(c(odds_ratio,      eval_epoch                            ), ~ as.numeric(as.character(.x))),
		   
		   across(c(sentence                                               ), ~ as.character(.x)),
		   
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   macro_verb_type    = fct_relevel(macro_verb_type, 'spray-type', 'load-type'),
		   micro_verb_type    = fct_relevel(micro_verb_type, 'particulate', 'goop',
		   												  	 'loading',     'stuffing'),
		   eval_verb          = fct_relevel(eval_verb, 'spray', 'shower', 'rub',  'dab',
		   									  		   'load',  'stock',  'pack', 'stuff'),
		   theme_det          = fct_relevel(theme_det, 'the', 'which', 'some', 'bare'),
		   goal_det           = fct_relevel(goal_det,  'the', 'which', 'a'),
		   tuning             = fct_relevel(tuning, 'spray/load theme-object spray active', 
		   											'spray/load goal-object spray active', 
		   											'spray/load theme-object load active', 
		   											'spray/load goal-object load active'),
		   structure          = fct_relevel(structure, 'theme-object', 'goal-object'),
		   training_verb      = fct_relevel(training_verb, 'spray', 'load'),
		   training_structure = fct_relevel(training_structure, 'theme-object', 'goal-object'))

sl.accuracies <- sl.accuracies |>
	mutate(across(c(s1,                 sentence_type,             predicted_arg,
		            predicted_role,     position_num_ref,          position_num_gen,
		            total_epochs,       min_epochs,                max_epochs, 
		            model_id,           eval_data,                 model_name, 
		            tuning,             masked_tuning_style,       patience,
		            delta,              epoch_criteria,            training_verb,
		            training_structure, macro_verb_type,           micro_verb_type,
		            eval_verb,          wh_question_type,          voice,
		            int_args_order_s1,  int_args_order_s2,         structure,          
		            a_moved_arg,        abar_moved_arg,            rc_type,
					training_macro_verb_type, training_micro_verb_type, random_seed,
					wh_moved_arg,    	clefted_arg,			   a_movement 			                              ), ~ as.factor(.x)),
	
		   across(c(masked,             strip_punct,               wh_movement,
		   			question,			polar_q,				   raising,
		   	        cleft,              v_part,                    rc,
		   	        abar_movement,      a_movement,                neg, 
		   	        alternation,        int_args_flipped,          part_shift,
		   	        int_args_crossover 										                  ), ~ as.logical(.x)),
		   
		   across(c(gen_given_ref,      both_correct,              ref_correct_gen_incorrect,
		   	        both_incorrect,     ref_incorrect_gen_correct, ref_correct,
		   	        ref_incorrect,      gen_correct,               gen_incorrect,
		   	        num_points,         specificity_.MSE.,         specificity_se,
		   	        eval_epoch                                                                ), ~ as.numeric(as.character(.x))),
		   
		   across(c(s1_ex,              s2_ex                                                 ), ~ as.character(.x)),
		   
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   macro_verb_type    = fct_relevel(macro_verb_type, 'spray-type', 'load-type'),
		   micro_verb_type    = fct_relevel(micro_verb_type, 'particulate', 'goop',
		   												     'loading',     'stuffing'),
		   eval_verb          = fct_relevel(eval_verb, 'spray', 'shower', 'rub',  'dab',
		   											   'load',  'stock',  'pack', 'stuff'),
		   tuning             = fct_relevel(tuning, 'spray/load theme-object spray active', 
		   											'spray/load goal-object spray active', 
		   											'spray/load theme-object load active', 
		   											'spray/load goal-object load active'),
		   structure          = fct_relevel(structure, 'theme-object', 'goal-object'),
		   training_verb      = fct_relevel(training_verb, 'spray', 'load'),
		   training_structure = fct_relevel(training_structure, 'theme-object', 'goal-object'))

sl.cossims <- sl.cossims |>
	mutate(across(c(predicted_arg,       token_id,      target_group,
					model_id,            model_name,    tuning,
					masked_tuning_style, total_epochs,  predicted_role,
					target_group_label,  patience,      delta,
					min_epochs,          max_epochs,    epoch_criteria,
					training_verb,       training_structure,
					training_macro_verb_type, training_micro_verb_type,
					random_seed										    ), ~ as.factor(.x)),
	
		   across(c(masked,              strip_punct                    ), ~ as.logical(.x)),
		   
		   across(c(cossim,              eval_epoch                     ), ~ as.numeric(as.character(.x))),
		   
		   across(c(token,                                              ), ~ as.character(.x)),
		   model_name         = fct_relevel(model_name, 'roberta', 'bert', 'distilbert'),
		   tuning             = fct_relevel(tuning, 'spray/load theme-object spray active', 
		   											'spray/load goal-object spray active', 
		   											'spray/load theme-object load active', 
		   											'spray/load goal-object load active'),
		   training_verb      = fct_relevel(training_verb, 'spray', 'load'),
		   training_structure = fct_relevel(training_structure, 'theme-object', 'goal-object'))
# for testing
dative.odds.ratios <- dative.odds.ratios |> 
	filter(!(sentence_type %like% '(amb)'),
		   training_verb != 'mail',
		   masked_tuning_style == 'always',
		   !strip_punct) |>
	droplevels() |> 
	mutate(ratio_name = toupper(as.character(ratio_name)),
		   ratio_name = case_when(ratio_name %like% 'RICKET/.*THAX' ~ 'RICKET/THAX',
		   						  ratio_name %like% 'THAX/.*RICKET' ~ 'THAX/RICKET'),
		   ratio_name = as.factor(ratio_name))

dative.accuracies <- dative.accuracies |> 
	filter(!(predicted_arg == 'any'), 
		   !(sentence_type %like% '(amb)'),
		   training_verb != 'mail',
		   masked_tuning_style == 'always',
		   !strip_punct) |>
	droplevels() |> 
	mutate(predicted_arg = toupper(predicted_arg),
		   predicted_arg = case_when(predicted_arg %like% 'RICKET' ~ 'RICKET',
		   							 predicted_arg %like% 'THAX'   ~ 'THAX'))

dative.cossims <- dative.cossims |>
	filter(training_verb != 'mail',
		   masked_tuning_style == 'always',
		   !strip_punct) |> 
	mutate(target_group  = as.character(target_group),
		   target_group  = case_when(target_group %like% 'most similar$' ~ target_group,
		   							 TRUE ~ toupper(target_group)),
		   target_group  = case_when(target_group %like% 'RICKET' ~ 'RICKET',
		   							 target_group %like% 'THAX'   ~ 'THAX',
		   							 TRUE ~ target_group),
		   target_group  = as.factor(target_group),
		   predicted_arg = toupper(as.character(predicted_arg)),
		   predicted_arg = case_when(predicted_arg %like% 'RICKET' ~ 'RICKET',
		   							 predicted_arg %like% 'THAX'   ~ 'THAX'),
		   predicted_arg = as.factor(predicted_arg))

sl.odds.ratios <- sl.odds.ratios |>
	filter(masked_tuning_style == 'always',
		   !strip_punct) |>
	mutate(ratio_name = toupper(as.character(ratio_name)),
		   ratio_name = case_when(ratio_name %like% 'GORX/.*THAX' ~ 'GORX/THAX',
		   						  ratio_name %like% 'THAX/.*GORX' ~ 'THAX/GORX'),
		   ratio_name = as.factor(ratio_name))

sl.accuracies <- sl.accuracies |> 
	filter(!(predicted_arg == 'any'),
		   masked_tuning_style == 'always',
		   !strip_punct) |>
	mutate(predicted_arg = toupper(as.character(predicted_arg)),
		   predicted_arg = case_when(predicted_arg %like% 'GORX' ~ 'GORX',
		   							 predicted_arg %like% 'THAX' ~ 'THAX'),
		   predicted_arg = as.factor(predicted_arg))

sl.cossims <- sl.cossims |>
	filter(masked_tuning_style == 'always',
		   !strip_punct) |> 
	mutate(target_group  = as.character(target_group),
		   target_group  = case_when(target_group %like% 'most similar$' ~ target_group,
		   							 TRUE ~ toupper(target_group)),
		   target_group  = case_when(target_group %like% 'GORX' ~ 'GORX',
		   							 target_group %like% 'THAX' ~ 'THAX',
		   							 TRUE ~ target_group),
		   target_group  = as.factor(target_group),
		   predicted_arg = toupper(as.character(predicted_arg)),
		   predicted_arg = case_when(predicted_arg %like% 'GORX' ~ 'GORX',
		   							 predicted_arg %like% 'THAX' ~ 'THAX'),
		   predicted_arg = as.factor(predicted_arg))