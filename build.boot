(set-env!
  :source-paths #{"src/java"}
  :resource-paths #{"src/clj"}
  :dependencies '[[org.clojure/clojure "1.9.0-alpha15"]
                  [nightlight "1.9.2" :scope "test"]
                  [org.nd4j/nd4j-api "0.8.0"]
                  [org.nd4j/nd4j-native "0.8.0"]
                  [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                  [hswick/jutsu.ai "0.0.7"]])

(require
  '[nightlight.boot :refer [nightlight]])

(deftask night []
  (comp
    (wait)
    (nightlight :port 4000)))

(deftask run []
  (with-pre-wrap fs
    (require 'shakespeare-rnn.core) 
    (let [main (resolve 'shakespeare-rnn.core/-main)]
      (main))
    fs))
