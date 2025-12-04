module ML1Utils

using Flux, Flux.Losses, Statistics, Random, Printf
using MLJ, MLJBase, MLJModelInterface, CategoricalArrays, StableRNGs
import LIBSVM

# Carga de modelos MLJ (solo una vez, a nivel de módulo)
const LIBSVM_SVC        = @load SVC pkg=LIBSVM verbosity=0
const MLJ_DTClassifier  = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
const MLJ_KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0
const MLJ_LogisticClf   = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

export oneHotEncoding, calculateMinMaxNormalizationParameters, normalizeMinMax!, normalizeMinMax,
       calculateZeroMeanNormalizationParameters, normalizeZeroMean!, normalizeZeroMean,
       classifyOutputs, accuracy, buildClassANN, trainClassANN, holdOut, confusionMatrix, printConfusionMatrix,
       crossvalidation, ANNCrossValidation, modelCrossValidation,
       VotingClassifier, trainClassEnsemble


function oneHotEncoding(feature::AbstractArray{<:Any,1},
        classes::AbstractArray{<:Any,1})
    @assert(all([in(value, classes) for value in feature]));
    
    numClasses = length(classes);
    @assert(numClasses>1)
    
    if (numClasses==2)
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end
    end
    return oneHot;
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
   minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters);
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
      normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters);
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end
    return ann;
end

# Unit3
function trainClassANN(topology::AbstractArray{<:Int,1},  
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                       maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
                       maxEpochsVal::Int=20, showText::Bool=false)

    Xtr, Ytr = trainingDataset
    Xva, Yva = validationDataset
    Xte, Yte = testDataset

    # Modelo y pérdida
    ann = buildClassANN(size(Xtr,2), topology, size(Ytr,2);
                        transferFunctions=transferFunctions)
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Flags
    hasval  = size(Yva,1) > 0
    hastest = size(Yte,1) > 0

    trainLosses = Float32[]; valLosses = Float32[]; testLosses = Float32[]
    trainingLoss = loss(ann, Xtr', Ytr')
    push!(trainLosses, trainingLoss)
    push!(valLosses,   hasval  ? loss(ann, Xva', Yva') : NaN32)
    push!(testLosses,  hastest ? loss(ann, Xte', Yte') : NaN32)

    # Optimizador
    opt = Flux.setup(Adam(learningRate), ann)

    # Early Stopping, si hay validación
    best_ann = deepcopy(ann)
    bestVal = Inf32
    noImprove = 0
    epoch = 0

    #Bucle de entrenamiento con posible Early Stopping (paciencia = maxEpochsVal)
    while (epoch < maxEpochs) && (trainingLoss > minLoss)
        Flux.train!(loss, ann, [(Xtr', Ytr')], opt)
        epoch += 1

        trainingLoss = loss(ann, Xtr', Ytr')
        push!(trainLosses, trainingLoss)

        if hasval
            v = loss(ann, Xva', Yva')
            push!(valLosses, v)
            if v < bestVal - 1f-7
                bestVal = v;
                best_ann = deepcopy(ann);
                noImprove = 0
            else
                noImprove += 1
            end
        else
            push!(valLosses, NaN32)
        end

        push!(testLosses, hastest ? loss(ann, Xte', Yte') : NaN32)

        if hasval && noImprove >= maxEpochsVal
            break
        end
        if showText && (epoch % 25 == 0)
            @info "Epoch $epoch: train_loss=$(trainLosses[end]) val_loss=$(valLosses[end]) test_loss=$(testLosses[end])"
        end
    end

    ann = hasval ? best_ann : ann
    return (ann, trainLosses, valLosses, testLosses)
end

# Unit3
function trainClassANN(topology::AbstractArray{<:Int,1},  
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
                       validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                       maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
                       maxEpochsVal::Int=20, showText::Bool=false)

    Xtr, ytr = trainingDataset
    Xva, yva = validationDataset
    Xte, yte = testDataset

    Ytr = reshape(ytr, :, 1)
    Yva = (length(yva) == 0) ? falses(0,1) : reshape(yva, :, 1)
    Yte = (length(yte) == 0) ? falses(0,1) : reshape(yte, :, 1)

    # Delegar a la versión matricial
    return trainClassANN(topology, (Xtr, Ytr);
                         validationDataset=(Xva, Yva),
                         testDataset=(Xte, Yte),
                         transferFunctions=transferFunctions,
                         maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
                         maxEpochsVal=maxEpochsVal, showText=showText)
end

# Unit3
function holdOut(N::Int, P::Real)
    @assert 0.0 < P < 1.0 "P debe estar en (0,1)"
    ntest = max(1, min(N-1, round(Int, N*P)))
    idx = randperm(N)
    testIdx  = idx[1:ntest]
    trainIdx = idx[(ntest+1):end]
    return (trainIdx, testIdx)
end

# Unit3
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert Pval ≥ 0 && Ptest ≥ 0 && Pval + Ptest < 1 "Pval+Ptest debe ser < 1"
    trainValIdx, testIdx = holdOut(N, Ptest)
    pval_rel = Pval / (1 - Ptest)
    tr_local, val_local = holdOut(length(trainValIdx), pval_rel)
    return (trainValIdx[tr_local], trainValIdx[val_local], testIdx)
end

# Unit4.1
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets)

    # Conteos de la matriz de confusión
    TP = count(outputs .& targets)
    TN = count(.!outputs .& .!targets)
    FP = count(outputs .& .!targets)
    FN = count(.!outputs .& targets)
    N  = TP + TN + FP + FN

    # Matriz 2x2 (rows: real [neg,pos], cols: predicted [neg,pos]) // (rows: [TN,FP] ; cols: [FN,TP]) 
    cm = Array{Int64,2}([TN FP; FN TP])

    # Métricas básicas
    acc = (TP + TN) / N
    err = (FP + FN) / N

    _safe(num, den) = den == 0 ? 0.0 : num/den

    # Valores por defecto
    sens = _safe(TP, TP + FN)      # recall
    spec = _safe(TN, TN + FP)
    ppv  = _safe(TP, TP + FP)      # precision
    npv  = _safe(TN, TN + FN)

    # Casos particulares:
    if TN == N
        # todos verdaderos negativos
        sens = 1.0; ppv = 1.0
    elseif TP == N
        # todos verdaderos positivos
        spec = 1.0; npv = 1.0
    end

    # F1
    f1 = (ppv + sens) == 0 ? 0.0 : 2 * (ppv * sens) / (ppv + sens)

    return (acc, err, sens, spec, ppv, npv, f1, cm)
end

# Unit4.1
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    preds = outputs .>= threshold
    return confusionMatrix(preds, targets)
end

# Unit4 : Función auxiliar para imprimir bonito matrices de confusión C x C
function _printConfusionMatrix(cm::AbstractMatrix{<:Integer};
                            row_labels::AbstractVector{<:AbstractString}=string.(1:size(cm,1)),
                            col_labels::AbstractVector{<:AbstractString}=string.(1:size(cm,2)),
                            title::AbstractString="Confusion matrix")

    @assert size(cm,1) == length(row_labels)
    @assert size(cm,2) == length(col_labels)

    # ancho por columna
    col_widths = [maximum(length.(string.(vcat(col_labels[j], cm[:,j]...)))) for j in 1:size(cm,2)]
    row_w = maximum(length.(row_labels))
    printstyled(title * "\n"; bold=true)
    println(" "^(row_w + 8) * "Predicted")

    # cabecera
    print(rpad("Actual", 8))
    print(rpad("", row_w))
    for j in 1:size(cm,2)
        print("  " * lpad(col_labels[j], col_widths[j]))
    end
    println()

    # filas
    for i in 1:size(cm,1)
        print(rpad("", 8) * rpad(row_labels[i], row_w))
        for j in 1:size(cm,2)
            print("  " * lpad(string(cm[i,j]), col_widths[j]))
        end
        println()
    end
    nothing
end

# Unit4 : Función auxiliar para imprimir métricas
function _printMetrics(acc, err, sens, spec, ppv, npv, f1; averaged_label::AbstractString="")
    println()
    if !isempty(averaged_label)
        println(averaged_label * " averages:")
    end
    @printf("Accuracy: %.4f\n", acc)
    @printf("Error rate: %.4f\n", err)
    @printf("Sensitivity (Recall): %.4f\n", sens)
    @printf("Specificity: %.4f\n", spec)
    @printf("Precision (PPV): %.4f\n", ppv)
    @printf("Negative Predictive V.: %.4f\n", npv)
    @printf("F1-score: %.4f\n", f1)
    nothing
end

# Unit4.1
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, err, sens, spec, ppv, npv, f1, cm) = confusionMatrix(outputs, targets)

    _printConfusionMatrix(cm; row_labels=["Neg","Pos"], col_labels=["Neg","Pos"])
    _printMetrics(acc, err, sens, spec, ppv, npv, f1)
end

# Unit4.1
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
                              targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    println("threshold = ", threshold)
    return printConfusionMatrix(outputs .>= threshold, targets)
end

# Unit4.2
function confusionMatrix(outputs::AbstractArray{Bool,2},
                         targets::AbstractArray{Bool,2};
                         weighted::Bool=true)

    @assert size(outputs) == size(targets)
    N, C = size(targets)
    @assert C != 2
    if C == 1
        return confusionMatrix(@view(outputs[:, 1]), @view(targets[:, 1]))     # Deriva al caso binario, usando columnas como vectores
    end

    # Métricas por clase
    sens = zeros(Float64, C)
    spec = zeros(Float64, C)
    ppv = zeros(Float64, C)
    npv = zeros(Float64, C)
    f1 = zeros(Float64, C)

    support = vec(sum(targets; dims=1))  # TP+FN por clase

    for c in 1:C
        if support[c] > 0 || any(@view outputs[:, c])
            # Llama a la confusionMatrix binaria sobre la columna c
            _acc, _err, s, sp, p, n, f, _cm = confusionMatrix(@view(outputs[:, c]), @view(targets[:, c]))
            sens[c] = s;  spec[c] = sp;  ppv[c] = p;  npv[c] = n;  f1[c] = f
        end
    end


    # Matriz de confusión CxC
    cm = [count(targets[:, i] .& outputs[:, j]) for i in 1:C, j in 1:C]

    # Agregación (macro o weighted)
    if weighted
        w = support
        wsum = max(sum(w), 1)
        sens_g = sum(w .* sens) / wsum
        spec_g = sum(w .* spec) / wsum
        ppv_g = sum(w .* ppv) / wsum
        npv_g = sum(w .* npv) / wsum
        f1_g = sum(w .* f1) / wsum
    else
        sens_g = mean(sens)
        spec_g = mean(spec)
        ppv_g = mean(ppv)
        npv_g = mean(npv)
        f1_g = mean(f1)
    end

    # Accuracy global
    acc = accuracy(outputs, targets)
    err = 1 - acc

    return (acc, err, sens_g, spec_g, ppv_g, npv_g, f1_g, cm)
end

# Unit4.2
function confusionMatrix(outputs::AbstractArray{<:Real,2},
                         targets::AbstractArray{Bool,2};
                         threshold::Real=0.5, weighted::Bool=true)
    
    preds = classifyOutputs(outputs; threshold=threshold)
    return confusionMatrix(preds, targets; weighted=weighted)
end

# Unit4.2
function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1},
                         classes::AbstractArray{<:Any,1};
                         weighted::Bool=true)
    
    # todas las etiquetas deben existir en 'classes'
    @assert all(in.(outputs, Ref(classes))) "Some outputs are not in classes"
    @assert all(in.(targets, Ref(classes))) "Some targets are not in classes"

    # One-hot según 'classes'
    Y  = oneHotEncoding(targets, classes)
    Ŷ  = oneHotEncoding(outputs, classes)
    
    return confusionMatrix(Ŷ, Y; weighted=weighted)
end

# Unit4.2
function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1};
                         weighted::Bool=true)
    
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end

# Unit4.2
function printConfusionMatrix(outputs::AbstractArray{Bool,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool=true)
    
    (acc, err, sens, spec, ppv, npv, f1, cm) = confusionMatrix(outputs, targets; weighted=weighted)

    C = size(cm,1)
    labels = string.(1:C)
    _printConfusionMatrix(cm; row_labels=labels, col_labels=labels, title="Confusion matrix (rows=Actual, cols=Predicted)")
    _printMetrics(acc, err, sens, spec, ppv, npv, f1; averaged_label = weighted ? "Weighted" : "Macro")
    nothing
end

# Unit4.2
function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool=true) # threshold ?
    
    (acc, err, sens, spec, ppv, npv, f1, cm) = confusionMatrix(outputs, targets; weighted=weighted)  # threshold ?

    C = size(cm,1)
    labels = string.(1:C)
    _printConfusionMatrix(cm; row_labels=labels, col_labels=labels, title="Confusion matrix (rows=Actual, cols=Predicted)")
    _printMetrics(acc, err, sens, spec, ppv, npv, f1; averaged_label = weighted ? "Weighted" : "Macro")
    nothing
end

# Unit4.2
function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1},
                              classes::AbstractArray{<:Any,1};
                              weighted::Bool=true)
    
    (acc, err, sens, spec, ppv, npv, f1, cm) = confusionMatrix(outputs, targets, classes; weighted=weighted)

    labels = string.(classes)
    _printConfusionMatrix(cm; row_labels=labels, col_labels=labels, title="Confusion matrix (rows=Actual, cols=Predicted)")
    _printMetrics(acc, err, sens, spec, ppv, npv, f1; averaged_label = weighted ? "Weighted" : "Macro")
    nothing
end

# Unit4.2
function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1};
                              weighted::Bool=true)
    
    classes = unique(vcat(targets, outputs))
    return printConfusionMatrix(outputs, targets, classes; weighted=weighted)
end


# Unit5
function crossvalidation(N::Int64, k::Int64)
    @assert k >= 2 "k must be at least 2"
    @assert N >= k "N must be at least k"

    base = repeat(collect(1:k), ceil(Int, N/k))
    folds = base[1:N]

    Random.shuffle!(folds)

    return folds
end

# Unit5
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    N = length(targets)
    @assert N >= k "N must be at least k"

    pos_idx = findall(targets)
    neg_idx = findall(.!targets)

    folds = zeros(Int, N)

    if !isempty(pos_idx)
        folds[pos_idx] = crossvalidation(length(pos_idx), k)
    end
    if !isempty(neg_idx)
        folds[neg_idx] = crossvalidation(length(neg_idx), k)
    end

    return folds
end

# Unit5
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N, C = size(targets)
    @assert N ≥ k "N must be at least k"

    # Caso binario
    if C == 1
        return crossvalidation(view(targets, :, 1), k)
    end

    folds = zeros(Int, N)

    # Bucle sobre clases
    for c in 1:C
        idx_c = findall(view(targets, :, c))
        if !isempty(idx_c)
            folds[idx_c] = crossvalidation(length(idx_c), k)
        end
    end

    return folds
end

# Unit5
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    Y = oneHotEncoding(targets)  # N×C (o N×1 en binario)
    return crossvalidation(Y, k)
end

# Unit5
function ANNCrossValidation(topology::AbstractArray{<:Int,1},
                            dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                            crossValidationIndices::Array{Int64,1};
                            numExecutions::Int=50,
                            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
                            validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)

    X, labels = dataset
    @assert size(X,1) == length(labels) "Instances (rows of X) must match labels length"
    N = size(X,1)
    
    # Clases y one-hot
    classes = unique(labels)
    Y = oneHotEncoding(labels, classes)
    C = max(1, size(Y, 2))

    # Número de folds
    k = maximum(crossValidationIndices)

    # Vectores para cada métrica
    accs = Float64[]; errs = Float64[]; senss = Float64[]; specs = Float64[]
    ppvs = Float64[]; npvs = Float64[]; f1s = Float64[]

    # Matriz de confusión global
    cm_global = zeros(Float64, (C == 1) ? (2,2) : (C,C))

    
    # Bucle k-fold
    for f in 1:k
        test_idx  = findall(crossValidationIndices .== f)
        train_idx = findall(crossValidationIndices .!= f)

        Xtr, Ytr = X[train_idx, :], Y[train_idx, :]
        Xte, Yte = X[test_idx,  :], Y[test_idx,  :]

        # Vectores para las métricas y la CM de cada ejecución dentro del fold
        acc_e = Float64[]; err_e = Float64[]; sens_e = Float64[]; spec_e = Float64[]
        ppv_e = Float64[];  npv_e = Float64[];  f1_e = Float64[]
        cms = Vector{Array{Float64,2}}()

        
        # Bucle de cada ejecución dentro del fold actual
        for e in 1:numExecutions
            # Validación interna
            if validationRatio > 0 && length(train_idx) > 1
                N_tr = length(train_idx)
                p_val_local = clamp(validationRatio * (N / N_tr), 1e-9, 1 - 1e-9)  # asegura 0 < p < 1
                
                tr_local, val_local = holdOut(N_tr, p_val_local)
                
                tr_idx  = train_idx[tr_local]
                val_idx = train_idx[val_local]
                
                Xtr2, Ytr2 = X[tr_idx,  :], Y[tr_idx,  :]
                Xva,  Yva  = X[val_idx, :], Y[val_idx, :]

                # Entrenamiento con validación y posible early stopping
                ann, _, _, _ = trainClassANN(topology, (Xtr2, Ytr2);
                                             validationDataset=(Xva, Yva),
                                             transferFunctions=transferFunctions,
                                             maxEpochs=maxEpochs, minLoss=minLoss,
                                             learningRate=learningRate,
                                             maxEpochsVal=maxEpochsVal,
                                             showText=showText)
            else
                # Entrenamiento sin validación (validationRatio=0)
                ann, _, _, _ = trainClassANN(topology, (Xtr, Ytr);
                                             transferFunctions=transferFunctions,
                                             maxEpochs=maxEpochs, minLoss=minLoss,
                                             learningRate=learningRate,
                                             maxEpochsVal=maxEpochsVal,
                                             showText=showText)
            end
            
            # Evaluación en test de la ejecución actual
            ŷ_test = ann(Xte')'
            a, e_rate, se, sp, p, n, f1, cm = confusionMatrix(ŷ_test, Yte)

            push!(acc_e, a);  push!(err_e, e_rate);  push!(sens_e, se);  push!(spec_e, sp)
            push!(ppv_e, p);  push!(npv_e, n);  push!(f1_e, f1)
            push!(cms, Float64.(cm))
        end

        # Media entre ejecuciones (una por fold)
        push!(accs,  mean(acc_e))
        push!(errs,  mean(err_e))
        push!(senss, mean(sens_e))
        push!(specs, mean(spec_e))
        push!(ppvs,  mean(ppv_e))
        push!(npvs,  mean(npv_e))
        push!(f1s,   mean(f1_e))

        # Matriz de confusión media del fold y acumulación en la global
        cm_mean_fold = if length(cms) == 1
            cms[1]
        else
            cm_stack = cat(cms...; dims=3)
            dropdims(mean(cm_stack, dims=3), dims=3)
        end
        cm_global .+= cm_mean_fold
    end

    # Agregación final (media y desviación entre folds), y matriz de confusión global
    return (
        (mean(accs),  std(accs)),
        (mean(errs),  std(errs)),
        (mean(senss), std(senss)),
        (mean(specs), std(specs)),
        (mean(ppvs),  std(ppvs)),
        (mean(npvs),  std(npvs)),
        (mean(f1s),   std(f1s)),
        cm_global
    )
end


# Unit6
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
                              dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
                              crossValidationIndices::Array{Int64,1})

    # Helper1 : leer "hp" (hyperparameters) aceptando clave String o Symbol
    _get(d::Dict, key::AbstractString, default) = haskey(d, key) ? d[key] :
                                                  haskey(d, Symbol(key)) ? d[Symbol(key)] : default
    # O estandarizar internamente a Symbol todas las claves del Dict
    #hp = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in modelHyperparameters)

    # Helper2 : mapeo del kernel SVM desde String/Symbol a LIBSVM.Kernel
    function _svm_kernel(kx)
        s = lowercase(String(kx))
        if     s == "linear" ; return LIBSVM.Kernel.Linear
        elseif s == "rbf" ; return LIBSVM.Kernel.RadialBasis
        elseif s == "sigmoid" ; return LIBSVM.Kernel.Sigmoid
        elseif s == "poly" || s == "polynomial"; return LIBSVM.Kernel.Polynomial
        else
            error("Kernel no soportado: $kx (usa \"linear\", \"rbf\", \"sigmoid\" o \"poly\")")
        end
    end

    # Helper3 : construir el modelo MLJ según el tipo
    function _build_model(kind::Symbol, hp::Dict)
        if kind in (:SVC, :SVM, :SVMClassifier)
            return _make_atom(:SVC, hp)
        elseif kind in (:DecisionTreeClassifier, :DecisionTree, :DT)
            return _make_atom(:DecisionTree, hp)
        elseif kind in (:KNNClassifier, :KNeighborsClassifier, :kNN, :kNNClassifier)
            return _make_atom(:KNN, hp)
        else
            error("Tipo de modelo no soportado para MLJ: $kind")
        end
    end

    # CASO ANN : delegar en función ANNCrossValidation del Unit5
    if modelType == :ANN
        @assert haskey(modelHyperparameters, "topology") || haskey(modelHyperparameters, :topology) "Falta 'topology' en modelHyperparameters para :ANN"
        topology = _get(modelHyperparameters, "topology", nothing)

        # parámetros opcionales si vienen 
        kwargs = Dict{Symbol,Any}()
        for ksym in (:numExecutions, :transferFunctions, :maxEpochs, :minLoss, :learningRate, :validationRatio, :maxEpochsVal, :showText)
            v = _get(modelHyperparameters, String(ksym), nothing)
            if v !== nothing
                kwargs[ksym] = v
            end
        end

        return ANNCrossValidation(topology, dataset, crossValidationIndices; (; kwargs...)...)
    end

    # CASO MLJ (SVM, DT o kNN)
    X, y_any = dataset
    
    y = string.(y_any)  # convertir etiquetas a String para evitar problemas
    classes = unique(y)
    k = maximum(crossValidationIndices)

    # vectores para las métricas por fold
    accs = Float64[]; errs = Float64[]; senss = Float64[]; specs = Float64[]
    ppvs = Float64[]; npvs = Float64[]; f1s  = Float64[]

    # CM global
    C = length(classes)
    cm_global = zeros(Int, C, C)

    for fold in 1:k
        train_mask = crossValidationIndices .!= fold
        test_mask  = .!train_mask

        Xtr, ytr = X[train_mask, :], y[train_mask]
        Xte, yte = X[test_mask,  :], y[test_mask]

        model = _build_model(modelType, modelHyperparameters)
        mach = machine(model, MLJ.table(Xtr), categorical(ytr))
        MLJ.fit!(mach, verbosity=0)

        yhat = MLJ.predict(mach, MLJ.table(Xte))

        ŷ_labels = if modelType in (:SVC, :SVM, :SVMClassifier)
            string.(yhat)
        elseif modelType in (:DecisionTreeClassifier, :DecisionTree, :DT, :KNNClassifier, :KNeighborsClassifier, :kNN, :kNNClassifier)
            string.(mode.(yhat))
        else
            error("Tipo de modelo no soportado: $modelType")
        end

        acc, err, sens, spec, ppv, npv, f1, cm = confusionMatrix(ŷ_labels, yte, classes)
        push!(accs, acc); push!(errs, err); push!(senss, sens); push!(specs, spec)
        push!(ppvs, ppv); push!(npvs, npv); push!(f1s, f1)
        cm_global .+= cm  
    end

    return (
        (Statistics.mean(accs),  Statistics.std(accs)),
        (Statistics.mean(errs),  Statistics.std(errs)),
        (Statistics.mean(senss), Statistics.std(senss)),
        (Statistics.mean(specs), Statistics.std(specs)),
        (Statistics.mean(ppvs),  Statistics.std(ppvs)),
        (Statistics.mean(npvs),  Statistics.std(npvs)),
        (Statistics.mean(f1s),   Statistics.std(f1s)),
        cm_global
    )
end


# Unit7
mutable struct VotingClassifier <: Probabilistic
    models::Vector{Probabilistic}
    voting::Symbol
    weights::Union{Nothing, Vector{Float64}}
end

# Unit7
function VotingClassifier(; models=Probabilistic[], voting::Symbol=:hard, weights=nothing)
    @assert voting in (:hard, :soft) "The only possible labels are :hard or :soft"

    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "Number of weights must match number of models"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        normalized_weights = Float64.(weights) ./ sum(weights)
    end

    return VotingClassifier(models, voting, normalized_weights)
end

# Unit7
function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    machs = [begin
        mm = machine(deepcopy(m), X, y)
        fit!(mm, verbosity=0)
        mm
    end for m in model.models]

    fitresults = (
        machines = machs,
        class_levels = collect(levels(y)),
        class_pool = CategoricalArrays.pool(y)
    )

    cache = nothing
    report = (n_models=length(model.models), voting=model.voting, weights=model.weights)
    return fitresults, cache, report
end

# Unit7
function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels

    predictions = [categorical(predict_mode(mach, Xnew), levels=class_levels) for mach in machines]
    n_samples = length(predictions[1])
    n_models = length(machines)
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

    ensemble_pred = similar(predictions[1])
    for i in 1:n_samples
        vote_counts = Dict{eltype(predictions[1][1]), Float64}()
        for (j, prediction) in enumerate(predictions)
            vote_counts[prediction[i]] = get(vote_counts, prediction[i], 0.0) + weights[j]
        end

        best_label = nothing
        best_score = -Inf
        for (lbl, sc) in vote_counts
            if sc > best_score
                best_score = sc
                best_label = lbl
            end
        end

        ensemble_pred[i] = best_label
    end

    return ensemble_pred
end

# Unit7
function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    machines     = fitresult.machines
    class_levels = fitresult.class_levels
    class_pool   = fitresult.class_pool

    if model.voting == :hard
        yhat = MLJModelInterface.predict_mode(model, fitresult, Xnew)
        yhat = categorical(yhat; levels=class_levels)

        return [MLJBase.UnivariateFinite(class_levels,
                                         [lvl == yhat[i] ? 1.0 : 0.0 for lvl in class_levels];
                                         pool=class_pool) for i in eachindex(yhat)]
    end

    all_predictions = [predict(mach, Xnew) for mach in machines]
    n_samples = length(all_predictions[1])
    n_models  = length(machines)
    n_classes = length(class_levels)
    weights   = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights

    avg_probs = zeros(n_samples, n_classes)
    for (w, prediction) in zip(weights, all_predictions)
        for i in 1:n_samples
            p_i = prediction[i]
            if p_i isa MLJBase.UnivariateFinite
                for (j, level) in enumerate(class_levels)
                    avg_probs[i, j] += w * pdf(p_i, level)
                end
            else
                for (j, level) in enumerate(class_levels)
                    avg_probs[i, j] += w * (p_i == level ? 1.0 : 0.0)
                end
            end
        end
    end

    for i in 1:n_samples
        s = sum(@view avg_probs[i, :])
        if s > 0
            @. avg_probs[i, :] = avg_probs[i, :] / s
        end
    end

    return [MLJBase.UnivariateFinite(class_levels, @view avg_probs[i, :]; pool=class_pool)
            for i in 1:n_samples]
end

# Unit7
MLJModelInterface.metadata_model(VotingClassifier,
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    load_path="VotingClassifier",
)

# Unit7
_labels_from_boolmatrix(Y::AbstractArray{Bool,2}) = begin
    N, C = size(Y)
    if C == 1
        return vec(@view Y[:, 1])
    else
        _, idx = findmax(Y, dims=2)
        return vec(idx)
    end
end

# Unit7
function _apply_params!(mdl, hp::Dict; svm_kind::Bool=false)
    if svm_kind
        if haskey(hp, :kernel) || haskey(hp, "kernel")
            kval = get(hp, :kernel, get(hp, "kernel", nothing))
            if kval !== nothing
                ks = lowercase(String(kval))
                kmapped = ks in ("rbf","radial","radialbasis","gaussian") ? LIBSVM.Kernel.RadialBasis :
                          ks in ("linear",) ? LIBSVM.Kernel.Linear :
                          ks in ("poly","polynomial") ? LIBSVM.Kernel.Polynomial :
                          ks in ("sigmoid","tanh") ? LIBSVM.Kernel.Sigmoid :
                          kval
                if hasproperty(mdl, :kernel)
                    setproperty!(mdl, :kernel, kmapped)
                end
            end
        end

        if hasproperty(mdl, :cost)
            if haskey(hp, "C")
                setproperty!(mdl, :cost, Float64(hp["C"]))
            elseif haskey(hp, :C)
                setproperty!(mdl, :cost, Float64(hp[:C]))
            end
        end
    end

    for (k, v) in hp
        pname = Symbol(k)
        (pname == :kernel || pname == :C) && continue
        if hasproperty(mdl, pname)
            setproperty!(mdl, pname, v)
        end
    end

    return mdl
end


# Unit7
function _make_atom(kind::Symbol, hp::Dict; force_probabilistic::Bool=false)
    if kind == :SVC
        mdl = LIBSVM_SVC()
        if force_probabilistic && hasproperty(mdl, :probability)
            mdl = LIBSVM_SVC(probability=true)
        end
        return _apply_params!(mdl, hp; svm_kind=true)

    elseif kind == :DecisionTree
        mdl = MLJ_DTClassifier()
        return _apply_params!(mdl, hp)

    elseif kind == :KNN
        mdl = MLJ_KNNClassifier()
        return _apply_params!(mdl, hp)

    elseif kind == :Logistic
        mdl = MLJ_LogisticClf()
        return _apply_params!(mdl, hp)

    else
        error("Base estimator not supported: $(kind)")
    end
end


# Unit7
function trainClassEnsemble(estimator::Symbol,
                            modelsHyperParameters::Dict,
                            ensembleHyperParameters::Dict,
                            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
                            kFoldIndices::Array{Int64,1})

    X, Y = trainingDataset
    @assert size(X,1) == size(Y,1) "X e Y deben tener el mismo número de filas."
    y = categorical(_labels_from_boolmatrix(Y))
    k = maximum(kFoldIndices)
    results = Vector{Float64}(undef, k)

    @load EnsembleModel pkg=MLJEnsembles verbosity=0
    for fold in 1:k
        test_idx = findall(kFoldIndices .== fold)
        train_idx = findall(kFoldIndices .!= fold)

        Xtr = MLJ.table(X[train_idx, :])
        Xte = MLJ.table(X[test_idx,  :])
        ytr = y[train_idx]
        yte = y[test_idx]

        atom = _make_atom(estimator, modelsHyperParameters)

        ens_kwargs = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in ensembleHyperParameters)
        ens_kwargs[:model] = atom
        ens_kwargs[:rng] = get(ens_kwargs, :rng, StableRNGs.StableRNG(123))

        ensemble = EnsembleModel(; ens_kwargs...)
        mach = MLJ.machine(ensemble, Xtr, ytr) |> fit!

        yhat = MLJ.predict_mode(mach, Xte)
        results[fold] = MLJ.accuracy(yhat, yte)
    end

    return (per_fold = results, mean = mean(results), std = std(results))
end

# Unit7
function trainClassEnsemble(estimators::AbstractArray{Symbol,1},
                            modelsHyperParameters::AbstractArray{Dict,1},
                            ensembleHyperParameters::Dict,
                            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
                            kFoldIndices::Array{Int64,1})

    @assert length(estimators) == length(modelsHyperParameters) "Un Dict de hyperparams por cada estimator"

    X, Y = trainingDataset
    @assert size(X,1) == size(Y,1) "X e Y deben tener el mismo número de filas."
    y = categorical(_labels_from_boolmatrix(Y))
    k = maximum(kFoldIndices)
    results = Vector{Float64}(undef, k)

    voting  = get(ensembleHyperParameters, :voting, get(ensembleHyperParameters, "voting", :hard))
    weights = get(ensembleHyperParameters, :weights, get(ensembleHyperParameters, "weights", nothing))

    for fold in 1:k
        test_idx  = findall(kFoldIndices .== fold)
        train_idx = findall(kFoldIndices .!= fold)

        Xtr = MLJ.table(X[train_idx, :])
        Xte = MLJ.table(X[test_idx,  :])
        ytr = y[train_idx]
        yte = y[test_idx]

        atoms = [_make_atom(estimators[i], modelsHyperParameters[i]; force_probabilistic=true)
                 for i in eachindex(estimators)]

        ensemble = VotingClassifier(models=atoms, voting=voting, weights=weights)
        mach = MLJ.machine(ensemble, Xtr, ytr) |> fit!

        yhat = MLJ.predict_mode(mach, Xte)
        results[fold] = MLJ.accuracy(yhat, yte)
    end

    return (per_fold = results, mean = mean(results), std = std(results))
end

end # module
