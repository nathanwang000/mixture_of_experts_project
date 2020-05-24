cluster:
	python generate_clusters.py --latent_dim 100 --ae_learning_rate 0.001 --ae_epochs 100 --num_clusters 3
careunit:
	python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --result_suffix careunit --repeats_allowed # result suffix needed to differentiate from custom
	python run_mortality_prediction.py --model_type GLOBAL --epochs 100 --repeats_allowed
careunit_test:
	python run_mortality_prediction.py --model_type GLOBAL --test_time --repeats_allowed
	python run_mortality_prediction.py --model_type MULTITASK --test_time --result_suffix careunit --repeats_allowed
custom:
	python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --cohorts custom --cohort_filepath test_clusters.npy --repeats_allowed
custom_test:
	python run_mortality_prediction.py --model_type GLOBAL --cohorts custom --cohort_filepath test_clusters.npy --test_time --repeats_allowed
	python run_mortality_prediction.py --model_type MULTITASK --cohorts custom --cohort_filepath test_clusters.npy --test_time --repeats_allowed
clean:
	rm -r data/mortality*
	rm *.npy
	rm -r clustering_models
	rm -r cluster_membership
	rm -r mortality_test
