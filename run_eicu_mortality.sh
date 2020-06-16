pcts=(5) #(30 1 5 10 20)
best_global_idx=(3) #(23 3 3 23 3)
for i in {0..4}
do
    pct=${pcts[i]}
    idx=${best_global_idx[i]}
    # echo $pct $idx
    python generate_settings_subset_mortality.py --global_model_fn ${idx}_pct${pct}_0_global_exp.m --train_data_subset_path eICU_data/mortality/pct_${pct}_train_indices/0.pkl --nc 10
done
