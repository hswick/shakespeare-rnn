(ns shakespeare-rnn.core
  (:import [datainput CharacterIterator Shakespeare]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.layers GravesLSTM$Builder]
           [org.nd4j.linalg.activations Activation]
           [org.deeplearning4j.nn.conf.layers RnnOutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.nn.conf BackpropType]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]))

(def lstm-layer-size 200)
(def mini-batch-size 32)
(def example-length 1000)
(def tbptt-length 50)
(def num-epochs 1)
(def generate-samples-every-n-mini-batches 10)
(def n-samples-to-generate 4)
(def n-characters-to-sample 300)
(def generation-initialization (String.))
(def rng (java.util.Random. 12345))

(def iter (Shakespeare/getShakespeareIterator mini-batch-size example-length))
(def n-out (.totalOutcomes iter))

(def conf (-> (NeuralNetConfiguration$Builder.)
              (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
              (.iterations 1)
              (.learningRate 0.1)
              (.seed 12345)
              (.regularization true)
              (.l2 0.001)
              (.weightInit (WeightInit/XAVIER))
              (.updater (Updater/RMSPROP))
              (.list)
              (.layer 0 (-> (GravesLSTM$Builder.)
                            (.nIn (.inputColumns iter))
                            (.nOut lstm-layer-size)
                            (.activation (Activation/TANH))
                            (.build)))
              (.layer 1 (-> (GravesLSTM$Builder.)
                            (.nIn lstm-layer-size)
                            (.nOut lstm-layer-size)
                            (.activation (Activation/TANH))
                            (.build)))
              (.layer 2 (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
                            (.activation (Activation/SOFTMAX))
                            (.nIn lstm-layer-size)
                            (.nOut n-out)
                            (.build)))
              (.backpropType (BackpropType/TruncatedBPTT))
              (.tBPTTForwardLength tbptt-length)
              (.tBPTTBackwardLength tbptt-length)
              (.pretrain false)
              (.backprop true)
              (.build)))

(def net (MultiLayerNetwork. conf))
(.init net)
(.setListeners net (list (ScoreIterationListener. 1)))

(def layers (.getLayers net))
(def total-num-params (atom 0))
(doseq [i (range 0 (count layers))]
  (let [num-params (.numParams (nth layers i))]
    (println (str "Number of pameters in layer " i ": " num-params))
    (reset! total-num-params (+ @total-num-params num-params))))
  
(def mini-batch-number (atom 1))

(doseq [i (range 0 num-epochs)]
  (println (str "Epoch " i))
  (while (.hasNext iter)
    (let [dataset (.next iter)]
      (.fit net dataset)
      (when (= 0 (mod @mini-batch-number generate-samples-every-n-mini-batches))
        (println "--------------------")
        (println (str "Completed " @mini-batch-number " minibatches of size " mini-batch-size "x" example-length " characters"))
        (println (str "Sampling characters from network given initialization '" generation-initialization "'"))
        (let [samples (Shakespeare/sampleCharactersFromNetwork nil net iter rng n-characters-to-sample n-samples-to-generate)]
          (doseq [j (range 0 (count samples))]
            (println (str "-----Sample " j " -------"))
            (println (nth samples j))
            (println)))))
    (reset! mini-batch-number (inc @mini-batch-number)))
  (.reset iter))
     
(defn -main []
  (println "We made our own Shakespeare!"))
