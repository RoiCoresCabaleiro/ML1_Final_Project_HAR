module RoiSubjectDisjoint

using Random, Statistics
using CSV, DataFrames
using MLJ, CategoricalArrays
using Printf
using Plots

using ..ML1Utils

export run_subject_disjoint_approach


function _find_col(names_vec, pattern::AbstractString)
    pl = lowercase(pattern)
    for n in names_vec
        if occursin(pl, lowercase(String(n)))
            return n
        end
    end
    error("No se ha encontrado ninguna columna que contenga '$pattern'")
end

function load_HAR_combined(train_path::AbstractString, test_path::AbstractString)
    df_train = CSV.read(train_path, DataFrame)
    df_test  = CSV.read(test_path,  DataFrame)
    df = vcat(df_train, df_test)

    cols = names(df)

    subj_col = _find_col(cols, "subject")
    act_candidates = [c for c in cols if occursin("activity", lowercase(String(c))) ||
                                     occursin("label",    lowercase(String(c)))]
    isempty(act_candidates) && error("No se encontró columna de actividad")
    act_col = act_candidates[1]

    feature_cols = setdiff(cols, [subj_col, act_col])

    X = Matrix{Float64}(df[:, feature_cols])
    y = Vector{String}(df[:, act_col])
    subjects = Vector{Int}(df[:, subj_col])

    return (X, y, subjects, feature_cols, act_col, subj_col)
end

function split_train_test1_test2(subjects::AbstractVector{<:Integer};
                                 test1_ratio::Real=0.30,
                                 test2_ratio::Real=0.05,
                                 seed::Int=1234)

    N = length(subjects)
    @assert test1_ratio > 0 && test2_ratio > 0 && test1_ratio + test2_ratio < 1

    uniq_subj = unique(subjects)
    # contadores de tuplas por sujeto
    counts = Dict(s => 0 for s in uniq_subj)
    for s in subjects
        counts[s] += 1
    end

    Random.seed!(seed)
    shuffled_subj = Random.shuffle(uniq_subj)

    # TEST1: sujetos completos (todas sus tuplas) hasta aproximar test1_ratio
    target_test1 = round(Int, test1_ratio * N)
    test1_subjects = Int[]
    n_test1 = 0
    for s in shuffled_subj
        push!(test1_subjects, s)
        n_test1 += counts[s]
        if n_test1 >= target_test1
            break
        end
    end
    test1_set = Set(test1_subjects)

    is_test1 = falses(N)
    for i in eachindex(subjects)
        if subjects[i] in test1_set
            is_test1[i] = true
        end
    end
    idx_test1 = findall(is_test1)

    # TEST2: de los restantes, tuplas aleatorias de diferentes sujetos siguiendo test2_ratio
    remaining_idx = findall(.!is_test1)
    target_test2  = round(Int, test2_ratio * N)

    Random.seed!(seed + 1)
    remaining_shuffled = Random.shuffle(remaining_idx)
    n_test2 = min(target_test2, length(remaining_shuffled))
    idx_test2 = sort(remaining_shuffled[1:n_test2])

    # TRAIN: tuplas sobrantes
    remaining_after_test2 = setdiff(remaining_idx, idx_test2)
    idx_train = sort(remaining_after_test2)

    return (idx_train, idx_test1, idx_test2, test1_subjects)
end

function subject_kfold_indices(subjects_train::AbstractVector{<:Integer},
                               k::Int; seed::Int=1234)
    uniq = unique(subjects_train)
    Random.seed!(seed)
    shuffled = Random.shuffle(uniq)

    # asignar fold en round-robin
    fold_per_subject = Dict{eltype(uniq),Int}()
    for (i, s) in enumerate(shuffled)
        fold_per_subject[s] = ((i - 1) % k) + 1
    end

    folds = Array{Int}(undef, length(subjects_train))
    for i in eachindex(subjects_train)
        folds[i] = fold_per_subject[subjects_train[i]]
    end
    return folds
end

function anova_f_scores(X::AbstractMatrix{<:Real}, y::Vector{String})
    N, D = size(X)
    classes = unique(y)
    C = length(classes)

    # Media global por feature
    μ_global = vec(mean(X; dims=1))

    # stats por clase
    n_c  = zeros(Float64, C)
    μ_c  = zeros(Float64, C, D)
    s2_c = zeros(Float64, C, D)

    for (ci, cls) in enumerate(classes)
        idx = findall(==(cls), y)
        n = length(idx)
        n_c[ci] = n
        if n == 0
            continue
        end
        Xc = X[idx, :]
        μ  = vec(mean(Xc; dims=1))
        μ_c[ci, :] .= μ

        # varianza muestral por feature (corregida)
        if n > 1
            s2 = vec(var(Xc; dims=1, corrected=true))
        else
            s2 = zeros(Float64, D)
        end
        s2_c[ci, :] .= s2
    end

    dfb = C - 1             # between
    dfw = N - C             # within
    F   = zeros(Float64, D)

    for j in 1:D
        num = 0.0
        den = 0.0
        for ci in 1:C
            nj = n_c[ci]
            if nj > 0
                num += nj * (μ_c[ci, j] - μ_global[j])^2
                den += (nj - 1) * s2_c[ci, j]
            end
        end
        msb = dfb > 0 ? num / dfb : 0.0
        msw = dfw > 0 ? den / dfw : 0.0
        F[j] = msw > 0 ? msb / msw : 0.0
    end

    return F
end

# 8 arquitecturas ANN (1–2 capas ocultas)
const ANN_TOPOLOGIES = [
    [64],
    [128],
    [256],
    [64, 32],
    [128, 64],
    [256, 128],
    [128, 128],
    [256, 256],
]

# 8 configuraciones SVM
const SVM_CONFIGS = [
    Dict("kernel"=>"linear", "C"=>0.1),
    Dict("kernel"=>"linear", "C"=>1.0),
    Dict("kernel"=>"linear", "C"=>10.0),
    Dict("kernel"=>"rbf",    "C"=>1.0, "gamma"=>0.01),
    Dict("kernel"=>"rbf",    "C"=>10.0, "gamma"=>0.01),
    Dict("kernel"=>"rbf",    "C"=>10.0, "gamma"=>0.001),
    Dict("kernel"=>"poly",   "C"=>1.0, "degree"=>2),
    Dict("kernel"=>"poly",   "C"=>1.0, "degree"=>3),
]

# 6 profundidades para árbol
const DT_CONFIGS = [
    Dict{String,Any}("max_depth" => d, "seed" => 1)
    for d in (3, 5, 7, 9, 11, 13)
]

# 6 valores de K para kNN
const KNN_CONFIGS = [
    Dict{String,Any}("K" => k)
    for k in (1, 3, 5, 7, 9, 11)
]


function run_cv_grid(modelType::Symbol,
                     hp_list::Vector{Dict{String,Any}},
                     X::AbstractArray{<:Real,2},
                     y::AbstractVector{<:Any},
                     fold_indices::Vector{Int})

    best_idx = 0
    best_acc = -Inf
    results = Vector{NamedTuple}(undef, length(hp_list))

    for (i, hp) in enumerate(hp_list)
        metrics = modelCrossValidation(modelType, hp, (X, y), fold_indices)
        (acc_mean, acc_std) = metrics[1]
        results[i] = (hp=hp, acc_mean=acc_mean, acc_std=acc_std, metrics=metrics)
        if acc_mean > best_acc
            best_acc = acc_mean
            best_idx = i
        end
    end

    return (results, best_idx)
end

function ann_hp_grid()
    [Dict{String,Any}(
        "topology"        => topo,
        "numExecutions"   => 1,
        "maxEpochs"       => 350,
        "learningRate"    => 0.005,
        "validationRatio" => 0.2,
        "maxEpochsVal"    => 10,
        "showText"        => false,
    ) for topo in ANN_TOPOLOGIES]
end

function svm_hp_grid()
    SVM_CONFIGS
end

function dt_hp_grid()
    DT_CONFIGS
end

function knn_hp_grid()
    KNN_CONFIGS
end

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

    mach = machine(mdl, MLJ.table(Xtr), categorical(string.(ytr)))
    fit!(mach, verbosity=0)

    yhat = predict(mach, MLJ.table(Xte))

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

# PIPELINE PRINCIPAL

function run_subject_disjoint_approach(; seed::Int = 1234)
    # A partir de aquí va el código que antes estaba suelto
    # desde `Random.seed!(1234)` hasta el final del archivo

    Random.seed!(seed)

    # 1) Cargar y combinar dataset
    train_path = "./datasets/train.csv"
    test_path  = "./datasets/test.csv"
    X_all, y_all, subj_all, feature_cols, act_col, subj_col =
        load_HAR_combined(train_path, test_path)

    N, D = size(X_all)
    println("Dataset combinado: N = $N muestras, D = $D features.")

    # 2) Triple split: Train / Test1 / Test2
    idx_train, idx_test1, idx_test2, test1_subjects =
        split_train_test1_test2(subj_all;
                                test1_ratio = 0.30,
                                test2_ratio = 0.05,
                                seed = seed)

    # Sujetos por split
    subj_train = subj_all[idx_train]
    subj_test1 = subj_all[idx_test1]

    n_subj_train = length(unique(subj_train))
    n_subj_test1 = length(unique(subj_test1))

    println("Train size = $(length(idx_train)) tuplas, $n_subj_train sujetos")
    println("Test1 size = $(length(idx_test1)) tuplas, $n_subj_test1 sujetos")
    println("Test2 size = $(length(idx_test2)) (tuplas aleatorias)")

    X_train = X_all[idx_train, :]
    y_train = y_all[idx_train]
    subj_train = subj_all[idx_train]

    X_test1 = X_all[idx_test1, :]
    y_test1 = y_all[idx_test1]

    X_test2 = X_all[idx_test2, :]
    y_test2 = y_all[idx_test2]

    # 3) Normalización Z-score
    norm_params = calculateZeroMeanNormalizationParameters(X_train)
    X_train_norm = Float32.(normalizeZeroMean(X_train, norm_params))
    X_test1_norm = Float32.(normalizeZeroMean(X_test1, norm_params))
    X_test2_norm = Float32.(normalizeZeroMean(X_test2, norm_params))

    # Índices de CV estratificados por sujeto
    k_folds = 10
    cv_indices = subject_kfold_indices(subj_train, k_folds; seed=seed)

    # Clases
    classes = unique(y_train)
    println("\nActivities: ", classes)
    println("\n")

    # Ranking ANOVA F-scores on training data
    F_scores  = anova_f_scores(X_train_norm, y_train)
    idx_sorted = sortperm(F_scores; rev=true)
    F_sorted   = F_scores[idx_sorted]

    # Gráfico: F-score vs. ranking de la feature
    ranks = 1:length(F_sorted)
    plot(ranks, F_sorted,
         xlabel = "Feature rank (sorted by decreasing F-score)",
         ylabel = "ANOVA F-score",
         legend = false,
         title  = "ANOVA F-scores over features")

    # Apply the selected Feature selection
    USE_FS  = true
    K_final = 300

    selected_idx = collect(1:D)

    if USE_FS
        @assert K_final ≤ length(idx_sorted)
        selected_idx = idx_sorted[1:K_final]

        X_train_norm = X_train_norm[:, selected_idx]
        X_test1_norm = X_test1_norm[:, selected_idx]
        X_test2_norm = X_test2_norm[:, selected_idx]

        println("Feature selection aplicado con K = $K_final features.")
        println("Dimensión tras FS: size(X_train_norm) = ", size(X_train_norm))
    else
        println("Feature selection desactivado (se usan las $D features originales).")
    end
    println("\n")

    #########################
    # 4) CV ANN / SVM / DT / kNN
    #########################
    println("\n")
    println("=== CV ANN (subject-wise) ===")
    ann_grid = ann_hp_grid()
    ann_results = Vector{NamedTuple}(undef, length(ann_grid))
    ann_best_idx = 0
    ann_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(ann_grid)
            training_time = @elapsed begin
                metrics = modelCrossValidation(:ANN, hp, (X_train_norm, y_train), cv_indices)
            end

            (acc_mean, acc_std) = metrics[1]
            ann_results[i] = (hp=hp, acc_mean=acc_mean, acc_std=acc_std, metrics=metrics)

            topo = hp["topology"]
            @printf("  [%d] topology=%s | acc=%.4f ± %.4f  |  %4d s\n", i, string(topo), acc_mean, acc_std, round(Int, training_time))

            if acc_mean > ann_best_acc
                ann_best_acc = acc_mean
                ann_best_idx = i
            end
        end
    end

    @printf(">> Mejor ANN idx = %d, acc = %.4f  |  %5d s\n\n",
            ann_best_idx, ann_best_acc, round(Int, total_training_time))

    println("\n")        
    println("=== CV SVM (subject-wise) ===")
    svm_grid = svm_hp_grid()
    svm_results = Vector{NamedTuple}(undef, length(svm_grid))
    svm_best_idx = 0
    svm_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(svm_grid)
            training_time = @elapsed begin
                metrics = modelCrossValidation(:SVM, hp,
                                               (X_train_norm, y_train),
                                               cv_indices)
            end

            (acc_mean, acc_std) = metrics[1]
            svm_results[i] = (hp=hp, acc_mean=acc_mean,
                              acc_std=acc_std, metrics=metrics)

            kernel = hp["kernel"]
            C      = hp["C"]
            gamma  = get(hp, "gamma", nothing)
            if gamma === nothing
                @printf("  [%d] kernel=%s, C=%.3g | acc=%.4f ± %.4f  |  %4d s\n",
                        i, String(kernel), C, acc_mean, acc_std,
                        round(Int, training_time))
            else
                @printf("  [%d] kernel=%s, C=%.3g, gamma=%.3g | acc=%.4f ± %.4f  |  %4d s\n",
                        i, String(kernel), C, gamma, acc_mean, acc_std,
                        round(Int, training_time))
            end

            if acc_mean > svm_best_acc
                svm_best_acc = acc_mean
                svm_best_idx = i
            end
        end
    end

    @printf(">> Mejor SVM idx = %d, acc = %.4f  |  %5d s\n\n",
            svm_best_idx, svm_best_acc, round(Int, total_training_time))

    println("\n")        
    println("=== CV Decision Tree (subject-wise) ===")
    dt_grid = dt_hp_grid()
    dt_results = Vector{NamedTuple}(undef, length(dt_grid))
    dt_best_idx = 0
    dt_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(dt_grid)
            training_time = @elapsed begin
                metrics = modelCrossValidation(:DecisionTree, hp,
                                               (X_train_norm, y_train),
                                               cv_indices)
            end

            (acc_mean, acc_std) = metrics[1]
            dt_results[i] = (hp=hp, acc_mean=acc_mean,
                             acc_std=acc_std, metrics=metrics)

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

    @printf(">> Mejor DT idx = %d, acc = %.4f  |  %5d s\n\n",
            dt_best_idx, dt_best_acc, round(Int, total_training_time))

    println("\n")
    println("=== CV kNN (subject-wise) ===")
    knn_grid = knn_hp_grid()
    knn_results = Vector{NamedTuple}(undef, length(knn_grid))
    knn_best_idx = 0
    knn_best_acc = -Inf

    total_training_time = @elapsed begin
        for (i, hp) in enumerate(knn_grid)
            training_time = @elapsed begin
                metrics = modelCrossValidation(:kNN, hp,
                                               (X_train_norm, y_train),
                                               cv_indices)
            end

            (acc_mean, acc_std) = metrics[1]
            knn_results[i] = (hp=hp, acc_mean=acc_mean,
                              acc_std=acc_std, metrics=metrics)

            K = hp["K"]
            @printf("  [%d] K=%2d | acc=%.4f ± %.4f  |  %4d s\n",
                    i, K, acc_mean, acc_std,
                    round(Int, training_time))

            if acc_mean > knn_best_acc
                knn_best_acc = acc_mean
                knn_best_idx = i
            end
        end
    end

    @printf(">> Mejor kNN idx = %d, acc = %.4f  |  %5d s\n\n",
            knn_best_idx, knn_best_acc, round(Int, total_training_time))

    #########################
    # 5) Entrenamiento final
    #########################


    println("\n")
    println("\n")
    # ANN
    ann_best_hp  = ann_results[ann_best_idx].hp
    ann_topology = ann_best_hp["topology"]
    println("Entrenando ANN final con topology = ", ann_topology)

    Y_train = oneHotEncoding(y_train, classes)
    ann, _, _, _ = trainClassANN(ann_topology, (X_train_norm, Y_train);
                                 maxEpochs    = ann_best_hp["maxEpochs"],
                                 learningRate = ann_best_hp["learningRate"],
                                 maxEpochsVal = ann_best_hp["maxEpochsVal"],
                                 showText     = false)

    Ŷ_ann_test1 = ann(X_test1_norm')'
    Ŷ_ann_test2 = ann(X_test2_norm')'

    function scores_to_labels(Ŷ, classes)
        Nte = size(Ŷ, 1)
        labels = Vector{String}(undef, Nte)
        for i in 1:Nte
            _, idx = findmax(Ŷ[i, :])
            labels[i] = string(classes[idx])
        end
        return labels
    end

    yhat_ann_test1 = scores_to_labels(Ŷ_ann_test1, classes)
    yhat_ann_test2 = scores_to_labels(Ŷ_ann_test2, classes)

    # SVM
    svm_best_hp = svm_results[svm_best_idx].hp
    println("Entrenando SVM final con hp = ", svm_best_hp)
    yhat_svm_test1 = train_mlj_model_and_predict(:SVM, svm_best_hp,
                                                 X_train_norm, y_train,
                                                 X_test1_norm)
    yhat_svm_test2 = train_mlj_model_and_predict(:SVM, svm_best_hp,
                                                 X_train_norm, y_train,
                                                 X_test2_norm)

    # kNN
    knn_best_hp = knn_results[knn_best_idx].hp
    println("Entrenando kNN final con hp = ", knn_best_hp)
    yhat_knn_test1 = train_mlj_model_and_predict(:kNN, knn_best_hp,
                                                 X_train_norm, y_train,
                                                 X_test1_norm)
    yhat_knn_test2 = train_mlj_model_and_predict(:kNN, knn_best_hp,
                                                 X_train_norm, y_train,
                                                 X_test2_norm)

    # DT
    dt_best_hp = dt_results[dt_best_idx].hp
    println("Entrenando DT final con hp = ", dt_best_hp)
    yhat_dt_test1 = train_mlj_model_and_predict(:DecisionTree, dt_best_hp,
                                                X_train_norm, y_train,
                                                X_test1_norm)
    yhat_dt_test2 = train_mlj_model_and_predict(:DecisionTree, dt_best_hp,
                                                X_train_norm, y_train,
                                                X_test2_norm)

    # Ensemble (hard voting: ANN + SVM + kNN)
    println("Preparando Ensemble (Hard-voting) con ANN + SVM + kNN")
    yhat_ens_test1 = majority_vote([yhat_ann_test1, yhat_svm_test1, yhat_knn_test1])
    yhat_ens_test2 = majority_vote([yhat_ann_test2, yhat_svm_test2, yhat_knn_test2])

    #########################
    # 6) Métricas y matrices
    #########################
    println("\n")
    println("\n")
    println("\n")
    println("======================================================")
    @printf("=            Test1 – Subject-disjoint                =\n")
    println("======================================================")

    acc_ann1, _, _, _, _, _, f1_ann1, _  = confusionMatrix(yhat_ann_test1, y_test1, classes)
    acc_svm1, _, _, _, _, _, f1_svm1, _  = confusionMatrix(yhat_svm_test1, y_test1, classes)
    acc_knn1, _, _, _, _, _, f1_knn1, _  = confusionMatrix(yhat_knn_test1, y_test1, classes)
    acc_dt1,  _, _, _, _, _, f1_dt1,  _  = confusionMatrix(yhat_dt_test1,  y_test1, classes)
    acc_ens1, err_ens1, sens_ens1,
    spec_ens1, ppv_ens1, npv_ens1,
    f1_ens1, _ = confusionMatrix(yhat_ens_test1, y_test1, classes)

    println("Resumen por modelo (accuracy / F1):")
    @printf("  ANN : acc=%.4f, F1=%.4f\n", acc_ann1, f1_ann1)
    @printf("  SVM : acc=%.4f, F1=%.4f\n", acc_svm1, f1_svm1)
    @printf("  kNN : acc=%.4f, F1=%.4f\n", acc_knn1, f1_knn1)
    @printf("  DT  : acc=%.4f, F1=%.4f\n", acc_dt1,  f1_dt1)
    @printf("  ENS : acc=%.4f, F1=%.4f\n", acc_ens1, f1_ens1)

    println("\nMatriz de confusión del ENSEMBLE en Test1:")
    printConfusionMatrix(yhat_ens_test1, y_test1)

    println("\n(Referencia) Matriz de confusión de la ANN en Test1:")
    printConfusionMatrix(yhat_ann_test1, y_test1)

    println("\n\n======================================================")
    @printf("=                  Test2 – Mixed                     =\n")
    println("======================================================")

    acc_ann2, _, _, _, _, _, f1_ann2, _  = confusionMatrix(yhat_ann_test2, y_test2, classes)
    acc_svm2, _, _, _, _, _, f1_svm2, _  = confusionMatrix(yhat_svm_test2, y_test2, classes)
    acc_knn2, _, _, _, _, _, f1_knn2, _  = confusionMatrix(yhat_knn_test2, y_test2, classes)
    acc_dt2,  _, _, _, _, _, f1_dt2,  _  = confusionMatrix(yhat_dt_test2,  y_test2, classes)
    acc_ens2, err_ens2, sens_ens2,
    spec_ens2, ppv_ens2, npv_ens2,
    f1_ens2, _ = confusionMatrix(yhat_ens_test2, y_test2, classes)

    println("Resumen por modelo (accuracy / F1):")
    @printf("  ANN : acc=%.4f, F1=%.4f\n", acc_ann2, f1_ann2)
    @printf("  SVM : acc=%.4f, F1=%.4f\n", acc_svm2, f1_svm2)
    @printf("  kNN : acc=%.4f, F1=%.4f\n", acc_knn2, f1_knn2)
    @printf("  DT  : acc=%.4f, F1=%.4f\n", acc_dt2,  f1_dt2)
    @printf("  ENS : acc=%.4f, F1=%.4f\n", acc_ens2, f1_ens2)

    println("\nMatriz de confusión del ENSEMBLE en Test2:")
    printConfusionMatrix(yhat_ens_test2, y_test2)

    println("\n(Referencia) Matriz de confusión de la ANN en Test2:")
    printConfusionMatrix(yhat_ann_test2, y_test2)

    return (
        classes = classes,
        selected_idx = selected_idx,
        ann_best_hp = ann_best_hp,
        svm_best_hp = svm_best_hp,
        dt_best_hp  = dt_best_hp,
        knn_best_hp = knn_best_hp,
        test1 = (acc = (ANN=acc_ann1, SVM=acc_svm1, KNN=acc_knn1, DT=acc_dt1, ENS=acc_ens1),
                 f1  = (ANN=f1_ann1, SVM=f1_svm1, KNN=f1_knn1, DT=f1_dt1, ENS=f1_ens1)),
        test2 = (acc = (ANN=acc_ann2, SVM=acc_svm2, KNN=acc_knn2, DT=acc_dt2, ENS=acc_ens2),
                 f1  = (ANN=f1_ann2, SVM=f1_svm2, KNN=f1_knn2, DT=f1_dt2, ENS=f1_ens2))
    )
end

end # module