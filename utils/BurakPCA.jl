module BurakPCA

using Random
using DelimitedFiles
using Statistics
using MLJ
using CategoricalArrays
using Plots

using ..ML1Utils
import LIBSVM

export run_pca_approach

# ---------------------------------------------------------
# Helper: majority vote (hard voting)
# ---------------------------------------------------------
function majority_vote(preds...)
    votes = Dict{eltype(preds[1]), Int}()
    for p in preds
        votes[p] = get(votes, p, 0) + 1
    end
    return argmax(votes)
end

# ---------------------------------------------------------
# Helper: plot confusion matrix as heatmap
# ---------------------------------------------------------
function plot_confusion_matrix(cm, class_labels, title_text)
    n = length(class_labels)

    p = heatmap(
        1:n, 1:n, cm;
        aspect_ratio = :equal,
        c            = :Blues,
        xlabel       = "Predicted",
        ylabel       = "Actual",
        title        = title_text,
        xticks       = (1:n, class_labels),
        yticks       = (1:n, class_labels),
        xrotation    = 45,
        size         = (500, 450),
        yflip        = true,
    )

    # Número en cada celda
    for i in 1:n
        for j in 1:n
            val = Int(round(cm[i, j]))
            annotate!(p, j, i, text(string(val), 9, :white, :center))
        end
    end

    return p
end

# ---------------------------------------------------------
# Helper: per-class analysis from confusion matrix
# ---------------------------------------------------------
function analyze_confusion_matrix(cm, class_labels, model_name)
    n_classes = length(class_labels)

    println("\n" * "="^80)
    println("Per-Class Performance Analysis - $model_name")
    println("="^80)
    println(
        "\n",
        rpad("Activity", 20),
        rpad("Precision", 12),
        rpad("Recall", 12),
        rpad("F1-Score", 12),
        "Support",
    )
    println("-"^80)

    for i in 1:n_classes
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        support = sum(cm[i, :])

        precision = tp > 0 ? tp / (tp + fp) : 0.0
        recall    = tp > 0 ? tp / (tp + fn) : 0.0
        f1        = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0

        println(
            rpad(string(class_labels[i]), 20),
            rpad(round(precision, digits = 4), 12),
            rpad(round(recall,    digits = 4), 12),
            rpad(round(f1,        digits = 4), 12),
            Int(round(support)),
        )
    end

    total_correct  = sum(cm[i, i] for i in 1:n_classes)
    total_samples  = sum(cm)
    overall_acc    = total_correct / total_samples

    println("-"^80)
    println("Overall Accuracy: ", round(overall_acc * 100, digits = 2), "%")
    println("Total Samples: ", Int(round(total_samples)))
    println("="^80)
end

# ---------------------------------------------------------
# Pipeline principal: PCA + CV + modelos + ensembles
# ---------------------------------------------------------
function run_pca_approach(; seed::Int = 1234)
    Random.seed!(seed)

    println("======================================================")
    println("        PCA-based approach (Burak) - seed = $seed     ")
    println("======================================================\n")

    # -----------------------------
    # Carga de datos
    # -----------------------------
    train_path = joinpath(@__DIR__, "..", "datasets", "train.csv")
    test_path  = joinpath(@__DIR__, "..", "datasets", "test.csv")

    train_data    = readdlm(train_path, ',', header = true)
    train_matrix  = train_data[1]
    train_headers = train_data[2]

    test_data     = readdlm(test_path, ',', header = true)
    test_matrix   = test_data[1]
    test_headers  = test_data[2]

    println("Train data shape: ", size(train_matrix))
    println("Test data shape: ", size(test_matrix))
    println("\nColumn headers (first 5): ", train_headers[1:5])

    # Inputs (todas menos la última) y outputs (última)
    X_train = Float32.(train_matrix[:, 1:end-1])
    y_train = train_matrix[:, end]

    X_test  = Float32.(test_matrix[:, 1:end-1])
    y_test  = test_matrix[:, end]

    # Clases
    classes = unique(vcat(y_train, y_test))
    println("Number of features: ", size(X_train, 2))
    println("Number of training samples: ", size(X_train, 1))
    println("Number of test samples: ", size(X_test, 1))
    println("\nClasses: ", classes)
    println("Number of classes: ", length(classes))

    println("Training set class distribution:")
    for class in classes
        count      = sum(y_train .== class)
        percentage = round(count / length(y_train) * 100, digits = 2)
        println("  $class: $count samples ($percentage%)")
    end

    println("\nTest set class distribution:")
    for class in classes
        count      = sum(y_test .== class)
        percentage = round(count / length(y_test) * 100, digits = 2)
        println("  $class: $count samples ($percentage%)")
    end

    # -----------------------------
    # Normalización Z-Score
    # -----------------------------
    normalization_params = calculateZeroMeanNormalizationParameters(X_train)
    X_train_normalized   = normalizeZeroMean(X_train, normalization_params)
    X_test_normalized    = normalizeZeroMean(X_test,  normalization_params)

    println("Standardization (Z-Score) completed.")
    println("Train Mean: ", round(mean(X_train_normalized), digits = 2))
    println("Train Std:  ", round(std(X_train_normalized),  digits = 2))

    # -----------------------------
    # PCA (95% var)
    # -----------------------------
    PCA = @load PCA pkg = MultivariateStats verbosity = 0

    pca_model   = PCA(variance_ratio = 0.95)
    pca_machine = machine(pca_model, MLJ.table(X_train_normalized))
    fit!(pca_machine, verbosity = 0)

    X_train_pca = MLJ.matrix(transform(pca_machine, MLJ.table(X_train_normalized)))
    X_test_pca  = MLJ.matrix(transform(pca_machine, MLJ.table(X_test_normalized)))

    println("PCA Transformation Results:")
    println("Original dimensions: ", size(X_train_normalized, 2))
    println("Reduced dimensions: ", size(X_train_pca, 2))
    println(
        "Dimensionality reduction: ",
        round((1 - size(X_train_pca, 2) / size(X_train_normalized, 2)) * 100, digits = 2),
        "%",
    )
    println("\nTrain set shape: ", size(X_train_pca))
    println("Test set shape: ", size(X_test_pca))

    # Comparación de distintos umbrales de varianza
    println("PCA Variance Threshold Comparison:")
    println("="^60)
    println("\nOriginal dimensions: 561 features")
    println("\nTesting different variance preservation thresholds:")

    variance_thresholds = [0.90, 0.95, 0.99]
    threshold_results   = []

    for variance_ratio in variance_thresholds
        pca_test      = PCA(variance_ratio = variance_ratio)
        pca_mach_test = machine(pca_test, MLJ.table(X_train_normalized))
        fit!(pca_mach_test, verbosity = 0)
        X_temp = MLJ.matrix(transform(pca_mach_test, MLJ.table(X_train_normalized)))

        n_components  = size(X_temp, 2)
        reduction_pct = round((1 - n_components / 561) * 100, digits = 2)

        println("\nVariance ratio = $(variance_ratio):")
        println("  Components retained: $n_components")
        println("  Dimensionality reduction: $reduction_pct%")
        println("  Features removed: $(561 - n_components)")

        push!(threshold_results, (ratio = variance_ratio, components = n_components, reduction = reduction_pct))
    end

    println("\n" * "="^60)
    println("Selected variance_ratio: 0.99")
    println("Rationale: Balances information preservation with dimensionality reduction")
    println("="^60)

    # -----------------------------
    # PCA 2D para visualización
    # -----------------------------
    pca_2d      = PCA(maxoutdim = 2)
    pca_2d_mach = machine(pca_2d, MLJ.table(X_train_normalized))
    fit!(pca_2d_mach, verbosity = 0)
    X_train_2d = MLJ.matrix(transform(pca_2d_mach, MLJ.table(X_train_normalized)))

    plot_fig = plot(
        title  = "PCA Projection (2D) - HAR Dataset",
        xlabel = "First Principal Component",
        ylabel = "Second Principal Component",
        legend = :outertopright,
    )

    colors = [:red, :blue, :green, :orange, :purple, :brown]
    for (i, class) in enumerate(classes)
        mask = y_train .== class
        scatter!(
            plot_fig,
            X_train_2d[mask, 1],
            X_train_2d[mask, 2];
            label      = string(class),
            markersize = 3,
            alpha      = 0.6,
            color      = colors[i],
        )
    end
    display(plot_fig)

    # -----------------------------
    # Etiquetas (MLJ y ANN)
    # -----------------------------
    all_labels   = categorical(vcat(y_train, y_test))
    y_train_cat  = all_labels[1:length(y_train)]
    y_test_cat   = all_labels[length(y_train) + 1:end]

    y_train_onehot = oneHotEncoding(y_train, classes)
    y_test_onehot  = oneHotEncoding(y_test,  classes)

    println("Labels prepared:")
    println("Categorical labels shape: ", length(y_train_cat), " (train), ", length(y_test_cat), " (test)")
    println("One-hot encoded shape: ", size(y_train_onehot), " (train), ", size(y_test_onehot), " (test)")

    # -----------------------------
    # Cross-validation estratificada
    # -----------------------------
    k_folds    = 10
    cv_indices = crossvalidation(y_train, k_folds)

    println("Cross-validation setup:")
    println("Number of folds: ", k_folds)
    println("CV indices length: ", length(cv_indices))
    println("\nSamples per fold:")
    for fold in 1:k_folds
        count = sum(cv_indices .== fold)
        println("  Fold $fold: $count samples")
    end

    # -----------------------------
    # Definición de arquitecturas ANN
    # -----------------------------
    ann_topologies = [
        [10],
        [20],
        [50],
        [100],
        [50, 25],
        [100, 50],
        [100, 50, 25],
        [200, 100],
    ]

    println("ANN architectures to test: ", length(ann_topologies))
    for (i, topology) in enumerate(ann_topologies)
        println("  Architecture $i: ", topology)
    end

    # -----------------------------
    # CV ANNs
    # -----------------------------
    println("\n" * "="^60)
    println("Training ANNs with Cross-Validation")
    println("="^60 * "\n")

    ann_results = Any[]

    for (i, topology) in enumerate(ann_topologies)
        println("\nTesting ANN Architecture $i: ", topology)

        training_time = @elapsed begin
            results = ANNCrossValidation(
                topology,
                (X_train_pca, y_train),
                cv_indices;
                numExecutions  = 5,
                maxEpochs      = 100,
                learningRate   = 0.01,
                validationRatio = 0.2,
                maxEpochsVal   = 10,
            )

            acc_mean, acc_std = results[1]
            err_mean, err_std = results[2]
            sens_mean, sens_std = results[3]
            spec_mean, spec_std = results[4]
            ppv_mean, ppv_std   = results[5]
            npv_mean, npv_std   = results[6]
            f1_mean,  f1_std    = results[7]

            println("  Accuracy:     $(round(acc_mean * 100, digits = 2))% ± $(round(acc_std * 100, digits = 2))%")
            println("  Error Rate:   $(round(err_mean * 100, digits = 2))% ± $(round(err_std * 100, digits = 2))%")
            println("  Sensitivity:  $(round(sens_mean, digits = 4)) ± $(round(sens_std, digits = 4))")
            println("  Specificity:  $(round(spec_mean, digits = 4)) ± $(round(spec_std, digits = 4))")
            println("  PPV:          $(round(ppv_mean,  digits = 4)) ± $(round(ppv_std,  digits = 4))")
            println("  NPV:          $(round(npv_mean,  digits = 4)) ± $(round(npv_std,  digits = 4))")
            println("  F1-Score:     $(round(f1_mean,   digits = 4)) ± $(round(f1_std,   digits = 4))")

            push!(ann_results, (
                topology = topology,
                accuracy = acc_mean,
                f1       = f1_mean,
                time     = training_time,
                results  = results,
            ))
        end

        println("  Training time: $(round(training_time, digits = 2))s")
    end

    best_ann = argmax([r.accuracy for r in ann_results])
    println("\n" * "="^60)
    println("Best ANN Architecture: ", ann_results[best_ann].topology)
    println("Best Accuracy: ", round(ann_results[best_ann].accuracy * 100, digits = 2), "%")
    println("Training Time: ", round(ann_results[best_ann].time, digits = 2), "s")
    println("="^60)

    # -----------------------------
    # CV SVM
    # -----------------------------
    svm_configs = [
        Dict("kernel" => "linear", "C" => 0.1),
        Dict("kernel" => "linear", "C" => 1.0),
        Dict("kernel" => "linear", "C" => 10.0),
        Dict("kernel" => "rbf",    "C" => 0.1, "gamma" => 0.01),
        Dict("kernel" => "rbf",    "C" => 1.0, "gamma" => 0.01),
        Dict("kernel" => "rbf",    "C" => 10.0, "gamma" => 0.01),
        Dict("kernel" => "poly",   "C" => 1.0, "degree" => 2),
        Dict("kernel" => "poly",   "C" => 1.0, "degree" => 3),
    ]

    println("SVM configurations to test: ", length(svm_configs))
    for (i, config) in enumerate(svm_configs)
        println("  Config $i: ", config)
    end

    println("\n" * "="^60)
    println("Training SVMs with Cross-Validation")
    println("="^60 * "\n")

    svm_results = Any[]

    for (i, config) in enumerate(svm_configs)
        println("\nTesting SVM Config $i: ", config)

        training_time = @elapsed begin
            results = modelCrossValidation(
                :SVC,
                config,
                (X_train_pca, y_train),
                cv_indices,
            )

            acc_mean, acc_std = results[1]
            f1_mean,  f1_std  = results[7]

            println("  Accuracy:     $(round(acc_mean * 100, digits = 2))% ± $(round(acc_std * 100, digits = 2))%")
            println("  F1-Score:     $(round(f1_mean, digits = 4)) ± $(round(f1_std, digits = 4))")

            push!(svm_results, (
                config   = config,
                accuracy = acc_mean,
                f1       = f1_mean,
                time     = training_time,
                results  = results,
            ))
        end

        println("  Training time: $(round(training_time, digits = 2))s")
    end

    best_svm = argmax([r.accuracy for r in svm_results])
    println("\n" * "="^60)
    println("Best SVM Configuration: ", svm_results[best_svm].config)
    println("Best Accuracy: ", round(svm_results[best_svm].accuracy * 100, digits = 2), "%")
    println("Training Time: ", round(svm_results[best_svm].time, digits = 2), "s")
    println("="^60)

    # -----------------------------
    # CV Decision Tree
    # -----------------------------
    dt_max_depths = [2, 4, 6, 8, 10, 15, 20, -1]  # -1 => unlimited

    println("Decision Tree max_depth values to test: ", length(dt_max_depths))
    println(dt_max_depths)

    println("\n" * "="^60)
    println("Training Decision Trees with Cross-Validation")
    println("="^60 * "\n")

    dt_results = Any[]

    for max_depth in dt_max_depths
        depth_str = max_depth == -1 ? "unlimited" : string(max_depth)
        println("\nTesting Decision Tree with max_depth=$depth_str")

        training_time = @elapsed begin
            results = modelCrossValidation(
                :DecisionTreeClassifier,
                Dict("max_depth" => max_depth, "seed" => 42),
                (X_train_pca, y_train),
                cv_indices,
            )

            acc_mean, acc_std = results[1]
            f1_mean,  f1_std  = results[7]

            println("  Accuracy:     $(round(acc_mean * 100, digits = 2))% ± $(round(acc_std * 100, digits = 2))%")
            println("  F1-Score:     $(round(f1_mean, digits = 4)) ± $(round(f1_std, digits = 4))")

            push!(dt_results, (
                max_depth = max_depth,
                accuracy  = acc_mean,
                f1        = f1_mean,
                time      = training_time,
                results   = results,
            ))
        end

        println("  Training time: $(round(training_time, digits = 2))s")
    end

    best_dt = argmax([r.accuracy for r in dt_results])
    best_depth_str = dt_results[best_dt].max_depth == -1 ? "unlimited" : string(dt_results[best_dt].max_depth)
    println("\n" * "="^60)
    println("Best Decision Tree max_depth: ", best_depth_str)
    println("Best Accuracy: ", round(dt_results[best_dt].accuracy * 100, digits = 2), "%")
    println("Training Time: ", round(dt_results[best_dt].time, digits = 2), "s")
    println("="^60)

    # -----------------------------
    # CV kNN
    # -----------------------------
    knn_k_values = [1, 3, 5, 7, 9, 11, 15, 21]

    println("kNN k values to test: ", length(knn_k_values))
    println(knn_k_values)

    println("\n" * "="^60)
    println("Training kNN with Cross-Validation")
    println("="^60 * "\n")

    knn_results = Any[]

    for k in knn_k_values
        println("\nTesting kNN with k=$k")

        training_time = @elapsed begin
            results = modelCrossValidation(
                :KNNClassifier,
                Dict("K" => k),
                (X_train_pca, y_train),
                cv_indices,
            )

            acc_mean, acc_std = results[1]
            f1_mean,  f1_std  = results[7]

            println("  Accuracy:     $(round(acc_mean * 100, digits = 2))% ± $(round(acc_std * 100, digits = 2))%")
            println("  F1-Score:     $(round(f1_mean, digits = 4)) ± $(round(f1_std, digits = 4))")

            push!(knn_results, (
                k        = k,
                accuracy = acc_mean,
                f1       = f1_mean,
                time     = training_time,
                results  = results,
            ))
        end

        println("  Training time: $(round(training_time, digits = 2))s")
    end

    best_knn = argmax([r.accuracy for r in knn_results])
    println("\n" * "="^60)
    println("Best kNN k value: ", knn_results[best_knn].k)
    println("Best Accuracy: ", round(knn_results[best_knn].accuracy * 100, digits = 2), "%")
    println("Training Time: ", round(knn_results[best_knn].time, digits = 2), "s")
    println("="^60)

    # -----------------------------
    # Visualización de CV (barplots)
    # -----------------------------
    ann_accs   = [r.accuracy * 100 for r in ann_results]
    ann_labels = [string(r.topology) for r in ann_results]

    p1 = bar(
        1:length(ann_accs), ann_accs;
        title   = "ANN Architectures - CV Accuracy",
        ylabel  = "Accuracy (%)",
        xlabel  = "Architecture",
        xticks  = (1:length(ann_labels), ann_labels),
        xrotation = 45,
        legend  = false,
        color   = :lightblue,
        ylim    = (90, 100),
    )

    svm_accs   = [r.accuracy * 100 for r in svm_results]
    svm_labels = [string(r.config["kernel"], " C=", r.config["C"]) for r in svm_results]

    p2 = bar(
        1:length(svm_accs), svm_accs;
        title   = "SVM Configurations - CV Accuracy",
        ylabel  = "Accuracy (%)",
        xlabel  = "Configuration",
        xticks  = (1:length(svm_labels), svm_labels),
        xrotation = 45,
        legend  = false,
        color   = :lightgreen,
        ylim    = (90, 100),
    )

    dt_accs   = [r.accuracy * 100 for r in dt_results]
    dt_labels = [r.max_depth == -1 ? "unlimited" : string(r.max_depth) for r in dt_results]

    p3 = bar(
        1:length(dt_accs), dt_accs;
        title   = "Decision Tree Depths - CV Accuracy",
        ylabel  = "Accuracy (%)",
        xlabel  = "Max Depth",
        xticks  = (1:length(dt_labels), dt_labels),
        legend  = false,
        color   = :orange,
    )

    knn_accs   = [r.accuracy * 100 for r in knn_results]
    knn_labels = [string("k=", r.k) for r in knn_results]

    p4 = bar(
        1:length(knn_accs), knn_accs;
        title   = "kNN k Values - CV Accuracy",
        ylabel  = "Accuracy (%)",
        xlabel  = "k Value",
        xticks  = (1:length(knn_labels), knn_labels),
        legend  = false,
        color   = :lightcoral,
        ylim    = (90, 100),
    )

    plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 800))

    println("\n" * "="^80)
    println("CROSS-VALIDATION RESULTS SUMMARY (PCA Approach)")
    println("="^80)

    println("\nBest ANN:")
    println("  Architecture: ", ann_results[best_ann].topology)
    println("  CV Accuracy:  ", round(ann_results[best_ann].accuracy * 100, digits = 2), "%")

    println("\nBest SVM:")
    println("  Configuration: ", svm_results[best_svm].config)
    println("  CV Accuracy:  ", round(svm_results[best_svm].accuracy * 100, digits = 2), "%")

    println("\nBest Decision Tree:")
    best_dt_depth_str = dt_results[best_dt].max_depth == -1 ? "unlimited" : string(dt_results[best_dt].max_depth)
    println("  Max Depth:   ", best_dt_depth_str)
    println("  CV Accuracy: ", round(dt_results[best_dt].accuracy * 100, digits = 2), "%")

    println("\nBest kNN:")
    println("  k value:     ", knn_results[best_knn].k)
    println("  CV Accuracy: ", round(knn_results[best_knn].accuracy * 100, digits = 2), "%")

    println("\n" * "="^80)
    println("\nTraining final models on full training set...\n")

    # -----------------------------
    # Entrenamiento final modelos
    # -----------------------------
    final_results = Dict{String, Dict{String, Any}}()

    # ANN final
    best_topology = ann_results[best_ann].topology
    println("Training final ANN with topology: ", best_topology)

    N_patterns = size(X_train_pca, 1)
    train_indices, val_indices = holdOut(N_patterns, 0.1)

    X_train_final = X_train_pca[train_indices, :]
    y_train_final = y_train_onehot[train_indices, :]

    X_val_final   = X_train_pca[val_indices, :]
    y_val_final   = y_train_onehot[val_indices, :]

    println("Data Split Completed:")
    println("  Final Training Samples: ", size(X_train_final, 1))
    println("  Validation Samples:     ", size(X_val_final, 1))

    final_ann, _, _, _ = trainClassANN(
        best_topology,
        (X_train_final, y_train_final);
        validationDataset = (X_val_final, y_val_final),
        maxEpochs         = 200,
        learningRate      = 0.01,
        maxEpochsVal      = 20,
        showText          = false,
    )

    y_pred_ann = final_ann(X_test_pca')'
    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_ann, y_test_onehot)

    final_results["ANN"] = Dict(
        "accuracy"          => acc,
        "f1"                => f1,
        "confusion_matrix"  => cm,
    )

    println("\nANN Test Results:")
    println("  Accuracy: ", round(acc * 100, digits = 2), "%")
    println("  F1-Score: ", round(f1, digits = 4))

    # SVM final
    best_svm_config = svm_results[best_svm].config
    println("\nTraining final SVM with config: ", best_svm_config)

    SVC = @load SVC pkg = LIBSVM

    kernel_str = best_svm_config["kernel"]
    kernel = kernel_str == "linear" ? LIBSVM.Kernel.Linear :
             kernel_str == "rbf"    ? LIBSVM.Kernel.RadialBasis :
             kernel_str == "poly"   ? LIBSVM.Kernel.Polynomial :
             LIBSVM.Kernel.RadialBasis

    svm_model = SVC(
        kernel = kernel,
        cost   = Float64(best_svm_config["C"]),
        gamma  = Float64(get(best_svm_config, "gamma", 0.01)),
        degree = Int32(get(best_svm_config, "degree", 3)),
    )

    svm_mach = machine(svm_model, MLJ.table(X_train_pca), y_train_cat)
    fit!(svm_mach, verbosity = 0)

    y_pred_svm        = predict(svm_mach, MLJ.table(X_test_pca))
    y_pred_svm_labels = string.(y_pred_svm)

    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_svm_labels, string.(y_test), classes)

    final_results["SVM"] = Dict(
        "accuracy"         => acc,
        "f1"               => f1,
        "confusion_matrix" => cm,
    )

    println("\nSVM Test Results:")
    println("  Accuracy: ", round(acc * 100, digits = 2), "%")
    println("  F1-Score: ", round(f1, digits = 4))

    # Decision Tree final
    best_dt_depth = dt_results[best_dt].max_depth
    println("\nTraining final Decision Tree with max_depth: ", best_dt_depth)

    DTClassifier = @load DecisionTreeClassifier pkg = DecisionTree
    dt_model = DTClassifier(max_depth = best_dt_depth, rng = Random.MersenneTwister(42))

    dt_mach = machine(dt_model, MLJ.table(X_train_pca), y_train_cat)
    fit!(dt_mach, verbosity = 0)

    y_pred_dt        = predict(dt_mach, MLJ.table(X_test_pca))
    y_pred_dt_labels = string.(mode.(y_pred_dt))

    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_dt_labels, string.(y_test), classes)

    final_results["DT"] = Dict(
        "accuracy"         => acc,
        "f1"               => f1,
        "confusion_matrix" => cm,
    )

    println("\nDecision Tree Test Results:")
    println("  Accuracy: ", round(acc * 100, digits = 2), "%")
    println("  F1-Score: ", round(f1, digits = 4))

    # kNN final
    best_k = knn_results[best_knn].k
    println("\nTraining final kNN with k: ", best_k)

    KNNClassifier = @load KNNClassifier pkg = NearestNeighborModels
    knn_model = KNNClassifier(K = best_k)

    knn_mach = machine(knn_model, MLJ.table(X_train_pca), y_train_cat)
    fit!(knn_mach, verbosity = 0)

    y_pred_knn        = predict(knn_mach, MLJ.table(X_test_pca))
    y_pred_knn_labels = string.(mode.(y_pred_knn))

    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_knn_labels, string.(y_test), classes)

    final_results["kNN"] = Dict(
        "accuracy"         => acc,
        "f1"               => f1,
        "confusion_matrix" => cm,
    )

    println("\nkNN Test Results:")
    println("  Accuracy: ", round(acc * 100, digits = 2), "%")
    println("  F1-Score: ", round(f1, digits = 4))

    # -----------------------------
    # Ensemble SVM + DT + kNN
    # -----------------------------
    println("\n" * "="^60)
    println("Creating Ensemble Model (Voting Classifier)")
    println("="^60 * "\n")

    pred_svm_str = y_pred_svm_labels
    pred_dt_str  = y_pred_dt_labels
    pred_knn_str = y_pred_knn_labels

    y_pred_ensemble = [
        majority_vote(pred_svm_str[i], pred_dt_str[i], pred_knn_str[i])
        for i in 1:length(y_test)
    ]

    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_ensemble, string.(y_test), classes)

    final_results["Ensemble"] = Dict(
        "accuracy"         => acc,
        "f1"               => f1,
        "confusion_matrix" => cm,
    )

    println("\nEnsemble Test Results:")
    println("  Accuracy: ", round(acc * 100, digits = 2), "%")
    println("  F1-Score: ", round(f1, digits = 4))

    # -----------------------------
    # Ensemble optimizado ANN + SVM + kNN
    # -----------------------------
    println("\n" * "="^60)
    println("Creating Optimized Ensemble (ANN + SVM + kNN)")
    println("="^60 * "\n")

    pred_ann_onehot = classifyOutputs(y_pred_ann)
    pred_ann_str = [classes[findfirst(pred_ann_onehot[i, :])] for i in 1:size(pred_ann_onehot, 1)]

    pred_svm_str = y_pred_svm_labels
    pred_knn_str = y_pred_knn_labels

    y_pred_ensemble = [
        majority_vote(pred_ann_str[i], pred_svm_str[i], pred_knn_str[i])
        for i in 1:length(y_test)
    ]

    acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(y_pred_ensemble, string.(y_test), classes)

    final_results["Ensemble"] = Dict(
        "accuracy"         => acc,
        "f1"               => f1,
        "confusion_matrix" => cm,
    )

    println("Ensemble Test Accuracy: ", round(acc * 100, digits = 2), "%")
    println("Ensemble F1-Score: ", round(f1, digits = 4))

    println("\n" * "="^80)
    println("FINAL TEST SET RESULTS - PCA APPROACH (Z-Score + 95% Variance)")
    println("="^80)

    for (model_name, results) in sort(collect(final_results), by = x -> x[2]["accuracy"], rev = true)
        acc = results["accuracy"]
        f1  = results["f1"]
        println(rpad(model_name, 15),
                "Accuracy: ", rpad(round(acc * 100, digits = 2), 6), "%   ",
                "F1-Score: ", round(f1, digits = 4))
    end
    println("="^80)

    # -----------------------------
    # Plots de resultados en test
    # -----------------------------
    model_names = ["ANN", "SVM", "DT", "kNN", "Ensemble"]
    accuracies  = [final_results[name]["accuracy"] * 100 for name in model_names]
    f1_scores   = [final_results[name]["f1"] for name in model_names]

    p_acc = bar(
        model_names,
        accuracies;
        title  = "Test Set Accuracy Comparison",
        ylabel = "Accuracy (%)",
        xlabel = "Model",
        legend = false,
        color  = :steelblue,
        ylim   = (0, 100),
    )
    for (i, accv) in enumerate(accuracies)
        annotate!(p_acc, i, accv + 2, text(string(round(accv, digits = 2), "%"), 8))
    end

    p_f1 = bar(
        model_names,
        f1_scores;
        title  = "Test Set F1-Score Comparison",
        ylabel = "F1-Score",
        xlabel = "Model",
        legend = false,
        color  = :coral,
        ylim   = (0, 1),
    )
    for (i, f1v) in enumerate(f1_scores)
        annotate!(p_f1, i, f1v + 0.05, text(string(round(f1v, digits = 4)), 8))
    end

    plot(p_acc, p_f1, layout = (1, 2), size = (1000, 400))

    # -----------------------------
    # Matrices de confusión
    # -----------------------------
    for (model_name, results) in final_results
        println("\n" * "="^60)
        println("Confusion Matrix - ", model_name)
        println("="^60)
        println(results["confusion_matrix"])
    end

    model_order = ["ANN", "SVM", "DT", "kNN", "Ensemble"]
    plots_cm    = Plots.Plot[]

    for model_name in model_order
        if haskey(final_results, model_name)
            cm = final_results[model_name]["confusion_matrix"]
            p  = plot_confusion_matrix(cm, string.(classes), "CM - $model_name")
            push!(plots_cm, p)
        end
    end

    if !isempty(plots_cm)
        plot(plots_cm..., layout = (2, 3), size = (1400, 900))
    end

    for model_name in model_order
        if haskey(final_results, model_name)
            cm = final_results[model_name]["confusion_matrix"]
            analyze_confusion_matrix(cm, classes, model_name)
        end
    end

    return final_results
end

end # module
