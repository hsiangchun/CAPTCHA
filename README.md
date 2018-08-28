# Overview
這個模型是以[cnn_lstm_ctc_ocr](https://github.com/weinman/cnn_lstm_ctc_ocr)為主來建立，並且加入了[cnn-for-captcha](https://github.com/lan2720/cnn-for-captcha)以及[re-deep-anpr](https://github.com/sapphirelin/re-deep-anpr)，前者用途為增加新的驗證碼來幫助val產生新的數據，並且補足先前資料不足的問題（inode不夠），後者的用途則為若驗整碼只在圖片的一小角時會增加自動的去捕捉圖片中可能有的驗證碼。


# Training
```Makefile
make mjsynth-download
make mjsynth-tfrecord
make train &
make monitor &
```
由於mjsynth dataset太大，所以我改用`CaptchaGenerator/new_captcha.py`來建立training data，並且再利用`src/make-tfrecord.py`來建立tfrecord檔案加快資料的讀取。若是能用mjsynth dataset則不需要多這些步驟。

# Testing
```Makefile
make test
```
透過`src/test.py`顯示目前結果。它會印出目前的 global_step, loss, label_acc, sequence_acc ，其中label_acc為字母正確率、sequence_acc為整個驗證碼全對的正確率。

# Validate 
```Makefile
make val
```
每次validate會產生自動5個英文混數字和5個純數字來驗證模型的正確率。它會印出 label , predicion , ratio ，其中label為預測目標、predicion為預測結果、ratio為預測目標與結果的差距。

```Makefile
make val_t
```
這部份為想要測試手上的數據而非自動產生的驗證碼，要注意的是在 `src` 中需要有 `final_val.txt`並且檔案之中要有正確的檔名。

```Makefile
make val_all  
```
這部份是將前面兩部份（`val`、`val_t`）一起執行，就不需要一直打兩次了。

# Problem
* 目前在大張圖片的部份會需要執行很久，因為圖片會被切割成非常多張小張的，儘管很明顯它不可能跟預測結果有關係，若是能改進這部份則在驗證的時候會變快許多。
* 在自動捕捉的部份其切割範圍(40\*100)為固定的，儘管會放大圖片來增加切割的機率，但仍然有可能會切割不出來，所以這部份也需要繼續改進。
