#!/bin/bash

function global_keras() {
    python run_mortality_prediction.py --model_type GLOBAL --epochs 100 --repeats_allowed --result_suffix $1
    python run_mortality_prediction.py --model_type GLOBAL --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function mtl_keras() {
    python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --repeats_allowed --result_suffix $1
    python run_mortality_prediction.py --model_type MULTITASK --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function separate_keras() {
    python run_mortality_prediction.py --model_type SEPARATE --epochs 100 --repeats_allowed --result_suffix $1
    python run_mortality_prediction.py --model_type SEPARATE --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function global_pytorch() {
    python moe.py --model_type GLOBAL --epochs 100 --repeats_allowed --result_suffix $1
    python moe.py --model_type GLOBAL --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function mtl_pytorch() {
    python moe.py --model_type MULTITASK --epochs 100 --repeats_allowed --result_suffix $1
    python moe.py --model_type MULTITASK --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function moe(){
    python moe.py --model_type MOE --epochs 100 --repeats_allowed --result_suffix $1
    python moe.py --model_type MOE --epochs 100 --repeats_allowed --test_time --result_suffix $1
}

function run() {
    # for i in {1..30};
    for i in {100,}; #{31..60};
    do
	j=0
	
	global_pytorch $i &
	let "j = $j + 1"
	pids[${j}]=$!
	
	moe $i &
	let "j = $j + 1"
	pids[${j}]=$!

	mtl_pytorch $i &
	let "j = $j + 1"
	pids[${j}]=$!
	
	# global_keras $i &
	# let "j = $j + 1"
	# pids[${j}]=$!

	# mtl_keras $i &
	# let "j = $j + 1"
	# pids[${j}]=$!

	# wait for finish
	for pid in ${pids[*]}; do
	    wait $pid
	done    
    done
}

run
