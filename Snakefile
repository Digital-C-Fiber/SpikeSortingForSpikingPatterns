configfile: "config.yaml"

DATASETS = list(config["datasets"].keys())
FEATURES = ["spdf_fv3", "w_raw", "spdf"]
CATEGORIES = ["detected", "background_all", "background_single"]
MODELS = ["svm", "one_class_svm", "xgboost"]

def get_recordings(wildcards):
    print(config["datasets"][wildcards.dataset])
    return config["datasets"][wildcards.dataset]["path"]


rule all:
    input:
        expand("results/scores/{model}/{feature}/{dataset}_results.csv",model=MODELS , dataset= DATASETS, feature=FEATURES),
        expand("datasets_test/nix/{dataset}.nix", dataset= DATASETS)


rule run_ocsvm:
    input:
        features_background_all=lambda wildcards: f"workflow/features/background_all/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        features_detected=lambda wildcards: f"workflow/features/detected/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        spikes_a =lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_spikes.pkl",
        ap_track_window_d=lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_track_window.pkl",
        ap_track_window_a=lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_track_window.pkl",
        ap_raw_a= lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_raw.pkl",
        ap_raw_d= lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_raw.pkl",
    output: 
        scores_oc="results/scores/one_class_svm/{feature}/{dataset}_results.csv",
        best_params_oc="results/scores/one_class_svm/param/{feature}/{dataset}_best_params.csv", 
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
        ground_truth_flag= lambda wildcards: config["datasets"][wildcards.dataset]["ground_truth_flag"],
    script: 
        "scripts/model/one_class_svm.py"


rule run_svm:
    input:
        features_background_all=lambda wildcards: f"workflow/features/background_all/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        features_detected=lambda wildcards: f"workflow/features/detected/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        spikes_a =lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_spikes.pkl",
        ap_track_window_d=lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_track_window.pkl",
        ap_track_window_a=lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_track_window.pkl",
        ap_raw_a= lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_raw.pkl",
        ap_raw_d= lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_raw.pkl",
    output: 
        scores_svm="results/scores/svm/{feature}/{dataset}_results.csv",
        best_params_svm="results/scores/svm/param/{feature}/{dataset}_best_params.csv", 
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
        ground_truth_flag= lambda wildcards: config["datasets"][wildcards.dataset]["ground_truth_flag"],
    script: 
        "scripts/model/svm_model.py"


rule run_xgb:
    input:
        features_background_all=lambda wildcards: f"workflow/features/background_all/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        features_detected=lambda wildcards: f"workflow/features/detected/{wildcards.feature}/{wildcards.dataset}_features.pkl",
        spikes_a =lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_spikes.pkl",
        ap_track_window_d=lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_track_window.pkl",
        ap_track_window_a=lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_track_window.pkl",
        ap_raw_a= lambda wildcards: f"workflow/dataframes/background_all/{wildcards.dataset}_ap_raw.pkl",
        ap_raw_d= lambda wildcards: f"workflow/dataframes/detected/{wildcards.dataset}_ap_raw.pkl",
    output: 
        scores_xgb="results/scores/xgboost/{feature}/{dataset}_results.csv",
        best_params_xgb="results/scores/xgboost/param/{feature}/{dataset}_best_params.csv", 
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
        ground_truth_flag= lambda wildcards: config["datasets"][wildcards.dataset]["ground_truth_flag"],
    script: 
        "scripts/model/xgboost_model.py"


rule feature_extraction:
    input:
        ap_window_iloc=lambda wildcards: f"workflow/dataframes/{wildcards.category}/{wildcards.dataset}_ap_window_iloc.pkl",
        ap_derivatives=lambda wildcards: f"workflow/dataframes/{wildcards.category}/{wildcards.dataset}_ap_derivatives.pkl",
        ap_track_window=lambda wildcards: f"workflow/dataframes/{wildcards.category}/{wildcards.dataset}_ap_track_window.pkl",
        ap_raw= lambda wildcards: f"workflow/dataframes/{wildcards.category}/{wildcards.dataset}_ap_raw.pkl",
        spikes=lambda wildcards: f"workflow/dataframes/{wildcards.category}/{wildcards.dataset}_spikes.pkl",
        raw_data=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_raw_data.pkl"
    output:
        features="workflow/features/{category}/{feature}/{dataset}_features.pkl",
        length_df="features/length/{category}/{feature}/{dataset}_length_features.csv"
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
    script:
        "scripts/feature_extraction.py"


rule prepare_data_classifier:
    input:
        spikes_detected= lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_spikes_detected.pkl",
        spikes=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_spikes.pkl",
        raw_data=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_raw_data.pkl"
    output:
        ap_window_iloc_d="workflow/dataframes/detected/{dataset}_ap_window_iloc.pkl",
        ap_derivatives_d="workflow/dataframes/detected/{dataset}_ap_derivatives.pkl",
        ap_track_window_d="workflow/dataframes/detected/{dataset}_ap_track_window.pkl",
        ap_raw_d="workflow/dataframes/detected/{dataset}_ap_raw.pkl",
        spikes_d="workflow/dataframes/detected/{dataset}_spikes.pkl",
        ap_window_iloc_a="workflow/dataframes/background_all/{dataset}_ap_window_iloc.pkl",
        ap_derivatives_a="workflow/dataframes/background_all/{dataset}_ap_derivatives.pkl",
        ap_raw_a="workflow/dataframes/background_all/{dataset}_ap_raw.pkl",
        ap_track_window_a="workflow/dataframes/background_all/{dataset}_ap_track_window.pkl",
        spikes_a="workflow/dataframes/background_all/{dataset}_spikes.pkl"
    params:
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
    script:
        "scripts/pre_processing_classification.py"
        

rule spike_detection:
    input:
        spikes=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_spikes.pkl",
        stimulations=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_stimulations.pkl",
        raw_data=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_raw_data.pkl"
    output:
        spikes_of_interest_file="workflow/spikes/{dataset}_spikes_of_interest.pkl",
        spikes_of_interest_df="workflow/dataframes/{dataset}_spikes_of_interest.pkl",
        spikes_detected_file= "workflow/detection/{dataset}_detection.csv",
        spikes_detected_df= "workflow/dataframes/{dataset}_spikes_detected.pkl",
        result_detection="results/detection/{dataset}_detection.csv",
        #template_threshold="workflow/templates/{dataset}_template_threshold.png"
    params:
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
        threshold=lambda wildcards: config["datasets"][wildcards.dataset]["threshold"],
    script:
        "scripts/detection_via_thresholding.py"


rule templates_and_SNR:
    input:
        raw_data=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_raw_data.pkl",
        ap_derivatives=lambda wildcards: f"workflow/pre_processing/{wildcards.dataset}_ap_derivatives.pkl",
        ap_track_window=lambda wildcards: f"workflow/pre_processing/{wildcards.dataset}_ap_track_window.pkl",
        ap_window_iloc=lambda wildcards: f"workflow/pre_processing/{wildcards.dataset}_ap_window_iloc.pkl",
        spikes=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_spikes.pkl",
        stimulations=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_stimulations.pkl"
    output:
        ap_templates="workflow/templates/{dataset}_ap_templates.pkl",
        template_figure="workflow/templates/{dataset}_templates.png",
        snr_figure="workflow/SNR/{dataset}_SNR.png",
        snr_file= "workflow/SNR/{dataset}_SNR.csv"
    params:
        track_of_interest_label=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", []),
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"]
    script:
        "scripts/templates_and_snr.py"


rule pre_process_data:
    input:
        raw_data=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_raw_data.pkl",
        stimulations=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_stimulations.pkl",
        spikes=lambda wildcards: f"workflow/dataframes/{wildcards.dataset}_spikes.pkl"
    output: 
        ap_window_iloc="workflow/pre_processing/{dataset}_ap_window_iloc.pkl",
        ap_derivatives="workflow/pre_processing/{dataset}_ap_derivatives.pkl",
        ap_track_window="workflow/pre_processing/{dataset}_ap_track_window.pkl",
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
    script:
        "scripts/pre_processing.py"


rule read_in_data:
    input:
        get_recordings
    output: 
        raw_data="workflow/dataframes/{dataset}_raw_data.pkl",
        stimulations="workflow/dataframes/{dataset}_stimulations.pkl",
        spikes="workflow/dataframes/{dataset}_spikes.pkl",
        nix_file= "datasets_test/nix/{dataset}.nix",
    params:
        name=lambda wildcards: config["datasets"][wildcards.dataset]["name"],
        path_dapsys=lambda wildcards: config["datasets"][wildcards.dataset]["path_dapsys"],
        path_nix=lambda wildcards: config["datasets"][wildcards.dataset]["path_nix"],
        time1=lambda wildcards: config["datasets"][wildcards.dataset]["time1"],
        time2=lambda wildcards: config["datasets"][wildcards.dataset]["time2"],
        tracks_of_interest=lambda wildcards: config["datasets"].get(wildcards.dataset, {}).get("track_of_interest", [])
    script:
        "scripts/read_in_data.py"