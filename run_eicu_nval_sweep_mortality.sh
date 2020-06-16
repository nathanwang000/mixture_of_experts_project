pct=10 #(30 1 5 10 20)
idx=23 #(23 3 3 23 3)
nvals=(0.01 0.05 0.10 0.2 0.3)
for i in {0..4} # rerun exp 7 and 4 for eicu/mortality
do
    pct_val=${nvals[i]}
    # echo $pct $idx $pct_val
    python generate_settings_subset_mortality.py --global_model_fn ${idx}_pct${pct}_0_global_exp.m --train_data_subset_path eICU_data/mortality/pct_${pct}_train_indices/0.pkl --pct_val $pct_val --nc 10 --result_dir_prefix "/data7/jiaxuan/"
done
