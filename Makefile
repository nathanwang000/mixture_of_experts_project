cluster:
	# python generate_clusters.py --latent_dim 100 --ae_learning_rate 0.001 --ae_epochs 100 --num_clusters 3
	# python cluster_moe.py --model_type GLOBAL
	python cluster_moe.py --model_type VAL_CURVE
	# python cluster_moe.py --model_type INPUT --pmt
	# python cluster_moe.py --model_type AE --ae_epochs 200
mtl_careunit:
	python run_mortality_prediction.py --model_type GLOBAL --epochs 100 --repeats_allowed
	python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --repeats_allowed
	python run_mortality_prediction.py --model_type SEPARATE --epochs 100 --repeats_allowed
mtl_custom:
	python run_mortality_prediction.py --model_type GLOBAL --epochs 100 --cohorts 'custom' --cohort_filepath test_clusters_embed100.npy --repeats_allowed --sample_weights
	python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --cohorts 'custom' --cohort_filepath test_clusters_embed100.npy --repeats_allowed --sample_weights
	python run_mortality_prediction.py --model_type SEPARATE --epochs 100 --cohorts 'custom' --cohort_filepath test_clusters_embed100.npy --repeats_allowed --sample_weights
	python moe.py --model_type MOE --test_time --repeats_allowed --epochs 100
mtl_careunit_test:
	python run_mortality_prediction.py --model_type GLOBAL --test_time --repeats_allowed
	python run_mortality_prediction.py --model_type MULTITASK --test_time --repeats_allowed
	python moe.py --model_type MOE --test_time --repeats_allowed
clean:
	rm -r data/mortality*
	rm *.npy
	rm -r clustering_models
	rm -r cluster_membership
	rm -r mortality_test
