(ns shakespeare-rnn.core
  (:require [jutsu.ai.core :as ai])
  (:import [datainput CharacterIterator Shakespeare]))

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

(def shakespeare-config
  [:optimization-algo :sgd
    :iterations 1
    :learning-rate 0.1
    :seed 12345
    :regularization true
    :l2 0.0001
    :weight-init :xavier
    :updater :rmsprop
    :layers [[:graves-lstm [:n-in (.inputColumns iter) :n-out lstm-layer-size :activation :tanh]]
             [:graves-lstm [:n-in lstm-layer-size :n-out lstm-layer-size :activation :tanh]]
             [:rnn-output :mcxent [:n-in lstm-layer-size :n-out n-out :activation :softmax]]]
    :backprop-type :truncated-bptt
    :tBPTT-backward-length tbptt-length
    :tBPTT-forward-length tbptt-length
    :pretrain false
    :backprop true])

(def n (ai/network shakespeare-config))

(def net (ai/initialize-net n))

(def layers (.getLayers net))
(def total-num-params (atom 0))
(doseq [i (range 0 (count layers))]
  (let [num-params (.numParams (nth layers i))]
    (println (str "Number of parameters in layer " i ": " num-params))
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
