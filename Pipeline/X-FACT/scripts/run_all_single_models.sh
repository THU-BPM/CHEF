

sources=(
ar_factuel_afp_com
bn_bengali_newschecker_in
bn_boombd_com
bn_dailyo_in
en_altnews_in
en_boomlive_in
en_factly_in
en_healthfeedback_org
en_indiatoday_in
en_newschecker_in
en_thelogicalindian_com
en_vishvasnews_com
es_animalpolitico_com
es_chequeado_com
es_colombiacheck_com
es_efe_com
es_maldita_es
es_newtral_es
fr_20minutes_fr
hi_aajtak_in
hi_altnews_in
hi_bbc_com
hi_vishvasnews_com
it_agi_it
it_pagellapolitica_it
ml_malayalam_factcrescendo_com
nl_nieuwscheckers_nl
no_faktisk_no
pa_vishvasnews_com
pl_sprawdzam_afp_com
pt_aosfatos_org
pt_apublica_org
pt_boatos_org
pt_bol_uol_com_br
pt_noticias_uol_com_br
pt_observador_pt
pt_piaui_folha_uol_com_br
pt_poligrafo_sapo_pt
sq_kallxo_com
sr_istinomer_rs
ta_tamil_factcrescendo_com
ta_youturn_in
te_factly_in
tr_teyit_org
ur_vishvasnews_com
zh_tfc-taiwan_org_tw
)


for src in "${sources[@]}"; do
    echo "Runnning for source $src"
    echo "python -u examples/text-classification/run_xfact.py --model_name_or_path bert-base-multilingual-cased --do_train --do_eval --data_dir data/xfact/most_data/  --learning_rate 2e-5 --num_train_epochs 10.0 --max_seq_length 128 --output_dir models/most_data/xfact_single_${src}_mbert_base/ --save_steps 10000 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --sources $src | tee logs/single_models/most_data_xfact_single_${src}_mbert_base.txt"
    python -u examples/text-classification/run_xfact.py --model_name_or_path bert-base-multilingual-cased --do_train --do_eval --data_dir data/xfact/most_data/  --learning_rate 2e-5 --num_train_epochs 10.0 --max_seq_length 128 --output_dir models/most_data/xfact_single_${src}_mbert_base/ --save_steps 10000 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --sources $src | tee logs/single_models/most_data_xfact_single_${src}_mbert_base.txt
    echo "Done Runnning for source $src"
done
