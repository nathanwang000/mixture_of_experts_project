pct=5 #(1 5 10 20 30)
idx=6 #(18 6 11 19 20)

nvals=(0.01 0.05 0.10 0.2 0.3)
for i in {0..4} # rerun exp 7 and 4 for eicu/arf
do
    pct_val=${nvals[i]}
    # echo $pct $idx $pct_val
    python generate_settings_subset_mortality.py --global_model_fn ${idx}_pct${pct}_0_global_exp.m --train_data_subset_path eICU_data/ARF_4.0h_download/pct_${pct}_train_indices/0.pkl --pct_val $pct_val --eicu_cohort ARF4 --nc 10
done
