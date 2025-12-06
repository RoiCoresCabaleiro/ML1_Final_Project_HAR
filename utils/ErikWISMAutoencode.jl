module ErikWISMAutoencode

using Random, Statistics, StatsBase
using CSV, DataFrames
using MLJ, CategoricalArrays
using MultivariateStats
using Printf
using Plots

using ..ML1Utils

export run_wism_autoencode_approach

# ------------------------------------------------------------------
#  Configuración general
# ------------------------------------------------------------------

const LDAModel = @load LDA pkg=MultivariateStats verbosity=0

# Topologías ANN relativamente ligeras (espacio LDA de baja dimensión)
const ANN_TOPOLOGIES = [
    [256],
    [128, 64],
    [256, 128],
    [128, 128],
    [256, 256],
    [256, 128, 64],
    [256, 256, 128],
    [256, 256, 257]
]

# Misma rejilla SVM/DT/kNN que en el enfoque subject-disjoint
const SVM_CONFIGS = [
    # Lineal
    Dict("kernel" => "linear", "C" => 0.1),
    Dict("kernel" => "linear", "C" => 1.0),
    Dict("kernel" => "linear", "C" => 10.0),
    # RBF
    Dict("kernel" => "rbf", "C" => 1.0,  "gamma" => 0.01),
    Dict("kernel" => "rbf", "C" => 10.0, "gamma" => 0.01),   # la que ya sabías que va bien
    Dict("kernel" => "rbf", "C" => 100.0,"gamma" => 0.01),
    # Poly
    Dict("kernel" => "poly", "C" => 1.0, "degree" => 2),
    Dict("kernel" => "poly", "C" => 1.0, "degree" => 3),
]

const DT_CONFIGS = [
    Dict{String,Any}("max_depth" => d, "seed" => 1)
    for d in (3, 5, 7, 9, 11, 13)
]

const KNN_CONFIGS = [
    Dict{String,Any}("K" => k)
    for k in (1, 3, 5, 7, 9, 11)
]

# ------------------------------------------------------------------
#  Normalización adaptativa (ajuste en train, aplicación en test)
# ------------------------------------------------------------------

function adaptive_normalize_fit(X::AbstractMatrix{<:Real})
    N, D = size(X)
    X_norm = Array{Float64}(undef, N, D)
    methods = Vector{Symbol}(undef, D)
    params  = Vector{NTuple{2,Float64}}(undef, D)

    for j in 1:D
        col = Float64.(X[:, j])
        sk = skewness(col)
        ku = kurtosis(col)

        # Heurística: distribución "no muy rara" → Z-score
        if abs(sk) < 1 && abs(ku - 3) < 3
            μ = mean(col)
            σ = std(col)
            σ = (σ == 0) ? 1e-6 : σ
            X_norm[:, j] = (col .- μ) ./ σ
            methods[j] = :zscore
            params[j]  = (μ, σ)
        else
            mn = minimum(col)
            mx = maximum(col)
            rng = mx - mn
            rng = (rng == 0) ? 1e-6 : rng
            X_norm[:, j] = (col .- mn) ./ rng
            methods[j] = :minmax
            params[j]  = (mn, mx)
        end
    end

    return X_norm, methods, params
end

function adaptive_normalize_apply(X::AbstractMatrix{<:Real},
                                  methods::Vector{Symbol},
                                  params::Vector{NTuple{2,Float64}})
    N, D = size(X)
    @assert D == length(methods) == length(params)
    X_norm = Array{Float64}(undef, N, D)

    for j in 1:D
        col = Float64.(X[:, j])
        m = methods[j]
        (a, b) = params[j]

        if m == :zscore
            μ, σ = a, b
            σ = (σ == 0) ? 1e-6 : σ
            X_norm[:, j] = (col .- μ) ./ σ
        else
            mn, mx = a, b
            rng = mx - mn
            rng = (rng == 0) ? 1e-6 : rng
            X_norm[:, j] = (col .- mn) ./ rng
        end
    end

    return X_norm
end

# ------------------------------------------------------------------
#  Helpers para CV y predicción MLJ
# ------------------------------------------------------------------

function ann_hp_grid()
    [Dict{String,Any}(
        "topology"        => topo,
        "numExecutions"   => 1,
        "maxEpochs"       => 600,
        "learningRate"    => 0.01,
        "validationRatio" => 0.1,
        "maxEpochsVal"    => 20,
        "showText"        => true,
    ) for topo in ANN_TOPOLOGIES]
end

svm_hp_grid()  = SVM_CONFIGS
dt_hp_grid()   = DT_CONFIGS
knn_hp_grid()  = KNN_CONFIGS

function train_mlj_model_and_predict(modelType::Symbol,
                                     hp::Dict{String,Any},
                                     Xtr::AbstractArray{<:Real,2},
                                     ytr::AbstractVector{<:Any},
                                     Xte::AbstractArray{<:Real,2})

    _get(h::Dict, key::AbstractString, default) =
        haskey(h, key) ? h[key] :
        haskey(h, Symbol(key)) ? h[Symbol(key)] : default

    mdl = nothing

    if modelType in (:SVC, :SVM, :SVMClassifier)
        mdl = ML1Utils._make_atom(:SVC, hp)

    elseif modelType in (:DecisionTreeClassifier, :DecisionTree, :DT)
        mdl = ML1Utils._make_atom(:DecisionTree, hp)

    elseif modelType in (:KNNClassifier, :KNeighborsClassifier, :kNN, :kNNClassifier)
        mdl = ML1Utils._make_atom(:KNN, hp)

    else
        error("Modelo no soportado: $modelType")
    end

    mach = MLJ.machine(mdl, MLJ.table(Xtr), categorical(string.(ytr)))
    MLJ.fit!(mach, verbosity = 0)

    yhat = MLJ.predict(mach, MLJ.table(Xte))

    if modelType in (:SVC, :SVM, :SVMClassifier)
        ŷ_labels = string.(yhat)           # SVC devuelve etiquetas directas
    else
        ŷ_labels = string.(mode.(yhat))    # DT/kNN devuelven UnivariateFinite
    end

    return ŷ_labels
end

function majority_vote(preds::Vector{Vector{String}})
    n_models = length(preds)
    @assert n_models > 0
    n_samples = length(preds[1])
    for p in preds
        @assert length(p) == n_samples
    end

    y_ens = Vector{String}(undef, n_samples)

    for i in 1:n_samples
        counts = Dict{String,Int}()
        for m in 1:n_models
            lab = preds[m][i]
            counts[lab] = get(counts, lab, 0) + 1
        end
        best_lab = ""
        best_cnt = -1
        for (lab, cnt) in counts
            if cnt > best_cnt
                best_cnt = cnt
                best_lab = lab
            end
        end
        y_ens[i] = best_lab
    end

    return y_ens
end

# ------------------------------------------------------------------
#  Enfoque completo Erik: WISM autoencoder + LDA + ANN/SVM/DT/kNN
# ------------------------------------------------------------------

function run_wism_autoencode_approach(; seed::Int = 1234)

    Random.seed!(seed)

    println("="^54)
    @printf("%-54s\n", "  WISM autoencoder + LDA approach (Erik)")
    println("="^54 * "\n")

    # --------------------------------------------------------------
    # 1) Cargar dataset WISMD autoencode
    # --------------------------------------------------------------
    data_path = joinpath(@__DIR__, "..", "datasets", "wismd_autoencode_64.csv")
    df = CSV.read(data_path, DataFrame)

    valid_classes = ["A", "B", "C", "D", "E"]
    mask = in.(df.activity, Ref(valid_classes))
    df = df[mask, :]

    emb_cols = ["emb$i" for i in 1:64]
    X_all = Matrix{Float64}(df[:, emb_cols])
    y_all = Vector{String}(df[:, :activity])

    N, D = size(X_all)
    classes = sort(unique(y_all))

    println("Dataset WISM autoencode:")
    @printf("  N samples = %d, D features = %d\n", N, D)
    println("  Classes: ", classes)

    counts = countmap(y_all)
    println("  Class distribution:")
    for c in classes
        n = get(counts, c, 0)
        @printf("    %s: %4d (%.2f%%)\n", c, n, 100 * n / N)
    end
    println()

    # --------------------------------------------------------------
    # 2) Hold-out 80/20 (train/test)
    # --------------------------------------------------------------
    Ptest = 0.2
    train_idx, test_idx = ML1Utils.holdOut(N, Ptest)

    X_train = X_all[train_idx, :]
    X_test  = X_all[test_idx, :]
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    Ntr, _ = size(X_train)
    Nte, _ = size(X_test)

    @printf("Train size: %d\n", Ntr)
    @printf("Test size : %d\n\n", Nte)

    # --------------------------------------------------------------
    # 3) Normalización adaptativa (ajustada en train)
    # --------------------------------------------------------------
    println("Adaptive normalization (train-based)...")
    X_train_norm, norm_methods, norm_params = adaptive_normalize_fit(X_train)
    X_test_norm = adaptive_normalize_apply(X_test, norm_methods, norm_params)

    global_mean = mean(X_train_norm)
    global_std  = std(vec(X_train_norm))

    @printf("  Global train mean ≈ %.4f\n", global_mean)
    @printf("  Global train std  ≈ %.4f\n\n", global_std)

    # --------------------------------------------------------------
    # 4) LDA sobre espacio normalizado
    # --------------------------------------------------------------
    println("Fitting LDA on normalized embeddings...")

    lda_model = LDAModel()
    lda_mach  = MLJ.machine(lda_model, MLJ.table(X_train_norm), categorical(y_train))
    MLJ.fit!(lda_mach, verbosity = 0)

    train_lda_tbl = MLJ.transform(lda_mach, MLJ.table(X_train_norm))
    test_lda_tbl  = MLJ.transform(lda_mach, MLJ.table(X_test_norm))

    X_train_lda = MLJ.matrix(train_lda_tbl)
    X_test_lda  = MLJ.matrix(test_lda_tbl)

    Ntr_lda, D_lda = size(X_train_lda)
    @printf("LDA space: %d samples, %d components\n\n", Ntr_lda, D_lda)

    # Re-escalado de las 4 componentes LDA (0-mean, unit-std)
    μ_lda = mean(X_train_lda, dims = 1)
    σ_lda = std(X_train_lda, dims = 1)
    σ_lda[σ_lda .== 0] .= 1e-6

    X_train_lda = (X_train_lda .- μ_lda) ./ σ_lda
    X_test_lda  = (X_test_lda  .- μ_lda) ./ σ_lda

    # Para Flux
    X_train_lda = Float32.(X_train_lda)
    X_test_lda  = Float32.(X_test_lda)

    # Visualización simple LDA 2D (si hay >=2 componentes)
    if D_lda ≥ 2
        println("Displaying LDA 2D projection (first two components)...")
        p_lda = plot(
            title  = "LDA Projection (2D) - WISM autoencode",
            xlabel = "LD1",
            ylabel = "LD2",
            legend = :outertopright,
        )

        palette = [:red, :blue, :green, :orange, :purple, :brown, :black]

        for (i, c) in enumerate(classes)
            mask_c = y_train .== c
            scatter!(
                p_lda,
                X_train_lda[mask_c, 1],
                X_train_lda[mask_c, 2];
                label      = string(c),
                markersize = 3,
                alpha      = 0.7,
                color      = palette[i],
            )
        end

        display(p_lda)
        println()
    end

    # --------------------------------------------------------------
    # 5) Cross-validation estratificada en espacio LDA
    # --------------------------------------------------------------
    println("Setting up stratified k-fold cross-validation on LDA features...")
    k_folds    = 10
    cv_indices = ML1Utils.crossvalidation(y_train, k_folds)

    println("  Number of folds: ", k_folds)
    println("  CV indices length: ", length(cv_indices))
    println()

    # --------------------------------------------------------------
    # 6) Grid de hiperparámetros y CV para ANN / SVM / DT / kNN
    # --------------------------------------------------------------
    # --- ANN ---
    println("=== CV ANN (WISM + LDA) ===")
    ann_grid = ann_hp_grid()
    ann_results = Vector{NamedTuple}(undef, length(ann_grid))
    ann_best_idx = 0
    ann_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(ann_grid)

            training_time = @elapsed begin
                metrics = ML1Utils.modelCrossValidation(
                    :ANN, hp, (X_train_lda, y_train), cv_indices)
                ann_results[i] = (hp = hp,
                                  acc_mean = metrics[1][1],
                                  acc_std  = metrics[1][2],
                                  metrics  = metrics)
            end

            acc_mean = ann_results[i].acc_mean
            acc_std  = ann_results[i].acc_std
            topo     = hp["topology"]

            @printf("  [%d] topology=%s | acc=%.4f ± %.4f  |  %4d s\n",
                    i, string(topo), acc_mean, acc_std,
                    round(Int, training_time))

            if acc_mean > ann_best_acc
                ann_best_acc = acc_mean
                ann_best_idx = i
            end
        end
    end

    @printf(">> Best ANN idx = %d, acc = %.4f  |  %5d s\n\n", ann_best_idx, ann_best_acc, round(Int, total_training_time))

    # --- SVM ---
    println("=== CV SVM (WISM + LDA) ===")
    svm_grid = svm_hp_grid()
    svm_results = Vector{NamedTuple}(undef, length(svm_grid))
    svm_best_idx = 0
    svm_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(svm_grid)

            training_time = @elapsed begin
                metrics = ML1Utils.modelCrossValidation(
                    :SVM, hp, (X_train_lda, y_train), cv_indices)
                svm_results[i] = (hp = hp,
                                  acc_mean = metrics[1][1],
                                  acc_std  = metrics[1][2],
                                  metrics  = metrics)
            end

            acc_mean = svm_results[i].acc_mean
            acc_std  = svm_results[i].acc_std
            kernel   = hp["kernel"]
            C        = hp["C"]
            gamma    = get(hp, "gamma", nothing)
            degree   = get(hp, "degree", nothing)

            if gamma === nothing && degree === nothing
                @printf("  [%d] kernel=%s, C=%.3g | acc=%.4f ± %.4f  |  %4d s\n",
                        i, String(kernel), C, acc_mean, acc_std,
                        round(Int, training_time))
            elseif gamma !== nothing
                @printf("  [%d] kernel=%s, C=%.3g, gamma=%.3g | acc=%.4f ± %.4f  |  %4d s\n",
                        i, String(kernel), C, gamma, acc_mean, acc_std,
                        round(Int, training_time))
            else
                @printf("  [%d] kernel=%s, C=%.3g, degree=%d | acc=%.4f ± %.4f  |  %4d s\n",
                        i, String(kernel), C, degree, acc_mean, acc_std,
                        round(Int, training_time))
            end

            if acc_mean > svm_best_acc
                svm_best_acc = acc_mean
                svm_best_idx = i
            end
        end
    end

    @printf(">> Best SVM idx = %d, acc = %.4f  |  %5d s\n\n",
            svm_best_idx, svm_best_acc, round(Int, total_training_time))

    # --- Decision Tree ---
    println("=== CV Decision Tree (WISM + LDA) ===")
    dt_grid = dt_hp_grid()
    dt_results = Vector{NamedTuple}(undef, length(dt_grid))
    dt_best_idx = 0
    dt_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(dt_grid)

            training_time = @elapsed begin
                metrics = ML1Utils.modelCrossValidation(
                    :DecisionTree, hp, (X_train_lda, y_train), cv_indices)
                dt_results[i] = (hp = hp,
                                 acc_mean = metrics[1][1],
                                 acc_std  = metrics[1][2],
                                 metrics  = metrics)
            end

            acc_mean = dt_results[i].acc_mean
            acc_std  = dt_results[i].acc_std
            max_depth = hp["max_depth"]

            @printf("  [%d] max_depth=%2d | acc=%.4f ± %.4f  |  %4d s\n",
                    i, max_depth, acc_mean, acc_std,
                    round(Int, training_time))

            if acc_mean > dt_best_acc
                dt_best_acc = acc_mean
                dt_best_idx = i
            end
        end
    end

    @printf(">> Best DT idx = %d, acc = %.4f  |  %5d s\n\n",
            dt_best_idx, dt_best_acc, round(Int, total_training_time))

    # --- kNN ---
    println("=== CV kNN (WISM + LDA) ===")
    knn_grid = knn_hp_grid()
    knn_results = Vector{NamedTuple}(undef, length(knn_grid))
    knn_best_idx = 0
    knn_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(knn_grid)

            training_time = @elapsed begin
                metrics = ML1Utils.modelCrossValidation(
                    :kNN, hp, (X_train_lda, y_train), cv_indices)
                knn_results[i] = (hp = hp,
                                  acc_mean = metrics[1][1],
                                  acc_std  = metrics[1][2],
                                  metrics  = metrics)
            end

            acc_mean = knn_results[i].acc_mean
            acc_std  = knn_results[i].acc_std
            K        = hp["K"]

            @printf("  [%d] K=%2d | acc=%.4f ± %.4f  |  %4d s\n",
                    i, K, acc_mean, acc_std,
                    round(Int, training_time))

            if acc_mean > knn_best_acc
                knn_best_acc = acc_mean
                knn_best_idx = i
            end
        end
    end

    @printf(">> Best kNN idx = %d, acc = %.4f  |  %5d s\n\n",
            knn_best_idx, knn_best_acc, round(Int, total_training_time))

    # --------------------------------------------------------------
    # 7) Entrenamiento final y evaluación en test
    # --------------------------------------------------------------
    println("=== Final training on full LDA train set and test evaluation ===\n")

    # One-hot para ANN
    Y_train = ML1Utils.oneHotEncoding(y_train, classes)
    Y_test  = ML1Utils.oneHotEncoding(y_test,  classes)

    # --- ANN final ---
    ann_best_hp  = ann_results[ann_best_idx].hp
    ann_topology = ann_best_hp["topology"]
    println("Training final ANN with topology = ", ann_topology)

    ann, _, _, _ = ML1Utils.trainClassANN(
        ann_topology, (X_train_lda, Y_train);
        maxEpochs    = ann_best_hp["maxEpochs"],
        learningRate = ann_best_hp["learningRate"],
        maxEpochsVal = ann_best_hp["maxEpochsVal"],
        showText     = false,
    )

    Y_pred_ann = ann(X_test_lda')'
    acc_ann, err_ann, sens_ann, spec_ann, ppv_ann, npv_ann, f1_ann, cm_ann =
        ML1Utils.confusionMatrix(Y_pred_ann, Y_test)

    println("\nANN Test Results:")
    @printf("  Accuracy: %.4f\n", acc_ann)
    @printf("  F1-score: %.4f\n", f1_ann)

    # Convertir preds ANN a etiquetas para el ensemble
    ann_onehot = ML1Utils.classifyOutputs(Y_pred_ann)
    y_pred_ann_labels = [classes[findfirst(ann_onehot[i, :])] for i in 1:size(ann_onehot, 1)]

    # --- SVM final ---
    svm_best_hp = svm_results[svm_best_idx].hp
    println("\nTraining final SVM with hp: ", svm_best_hp)
    y_pred_svm = train_mlj_model_and_predict(:SVM, svm_best_hp,
                                             X_train_lda, y_train,
                                             X_test_lda)

    acc_svm, err_svm, sens_svm, spec_svm, ppv_svm, npv_svm, f1_svm, cm_svm =
        ML1Utils.confusionMatrix(y_pred_svm, y_test, classes)

    println("\nSVM Test Results:")
    @printf("  Accuracy: %.4f\n", acc_svm)
    @printf("  F1-score: %.4f\n", f1_svm)

    # --- DT final ---
    dt_best_hp = dt_results[dt_best_idx].hp
    println("\nTraining final Decision Tree with hp: ", dt_best_hp)
    y_pred_dt = train_mlj_model_and_predict(:DecisionTree, dt_best_hp,
                                            X_train_lda, y_train,
                                            X_test_lda)

    acc_dt, err_dt, sens_dt, spec_dt, ppv_dt, npv_dt, f1_dt, cm_dt =
        ML1Utils.confusionMatrix(y_pred_dt, y_test, classes)

    println("\nDecision Tree Test Results:")
    @printf("  Accuracy: %.4f\n", acc_dt)
    @printf("  F1-score: %.4f\n", f1_dt)

    # --- kNN final ---
    knn_best_hp = knn_results[knn_best_idx].hp
    println("\nTraining final kNN with hp: ", knn_best_hp)
    y_pred_knn = train_mlj_model_and_predict(:kNN, knn_best_hp,
                                             X_train_lda, y_train,
                                             X_test_lda)

    acc_knn, err_knn, sens_knn, spec_knn, ppv_knn, npv_knn, f1_knn, cm_knn =
        ML1Utils.confusionMatrix(y_pred_knn, y_test, classes)

    println("\nkNN Test Results:")
    @printf("  Accuracy: %.4f\n", acc_knn)
    @printf("  F1-score: %.4f\n", f1_knn)

    # --- Ensemble simple (ANN + SVM + kNN) ---
    println("\n=== Simple ensemble (majority vote: ANN + SVM + kNN) ===")
    y_pred_ens = majority_vote([y_pred_ann_labels, y_pred_svm, y_pred_knn])

    acc_ens, err_ens, sens_ens, spec_ens, ppv_ens, npv_ens, f1_ens, cm_ens =
        ML1Utils.confusionMatrix(y_pred_ens, y_test, classes)

    println("Ensemble Test Results:")
    @printf("  Accuracy: %.4f\n", acc_ens)
    @printf("  F1-score: %.4f\n", f1_ens)

    println("\nDone.\n")

    return (
        classes = classes,
        ann = (
            best_hp_cv   = ann_best_hp,
            cv_results   = ann_results,
            acc_test     = acc_ann,
            f1_test      = f1_ann,
            cm_test      = cm_ann,
        ),
        svm = (
            best_hp_cv   = svm_best_hp,
            cv_results   = svm_results,
            acc_test     = acc_svm,
            f1_test      = f1_svm,
            cm_test      = cm_svm,
        ),
        dt = (
            best_hp_cv   = dt_best_hp,
            cv_results   = dt_results,
            acc_test     = acc_dt,
            f1_test      = f1_dt,
            cm_test      = cm_dt,
        ),
        knn = (
            best_hp_cv   = knn_best_hp,
            cv_results   = knn_results,
            acc_test     = acc_knn,
            f1_test      = f1_knn,
            cm_test      = cm_knn,
        ),
        ensemble = (
            acc_test     = acc_ens,
            f1_test      = f1_ens,
            cm_test      = cm_ens,
        ),
    )
end

end # module
