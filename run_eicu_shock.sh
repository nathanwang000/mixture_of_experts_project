pcts=(1 5 10 20 30)
best_global_idx=(29 17 23 3 16)
for i in {0..4}
do
    pct=${pcts[i]}
    idx=${best_global_idx[i]}
    # echo $pct $idx
    python generate_settings_subset_shock.py --global_model_fn ${idx}_pct${pct}_0_global_exp.m --train_data_subset_path eICU_data/Shock_4.0h_download/pct_${pct}_train_indices/0.pkl --eicu_cohort Shock4 --nc 10
done
