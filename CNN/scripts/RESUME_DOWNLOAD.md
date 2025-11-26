# 📥 Resume UVH-26 Dataset Download

Your download was rate-limited at **1.7GB (1,566 files)** out of **~27GB (26,652 files)**

## 🚀 Quick Resume (2 steps):

### Step 1: Get Your Token
1. Open: https://huggingface.co/settings/tokens
2. Sign up/Login (free)
3. Click **"Create new token"**
4. Name: `dataset-download`
5. Type: **Read**
6. Copy the token (starts with `hf_...`)

### Step 2: Run with Token
```bash
cd CNN/scripts
HF_TOKEN="your_token_here" python3 download_uvh26.py
```

**Replace `your_token_here` with your actual token!**

---

## 📊 Check Progress Anytime:

```bash
# Size downloaded
du -sh CNN/datasets/UVH-26/

# Number of files
find CNN/datasets/UVH-26/ -type f | wc -l

# Watch download progress
watch -n 10 'du -sh CNN/datasets/UVH-26/ && find CNN/datasets/UVH-26/ -type f | wc -l'
```

---

## ✅ What's Already Downloaded:

- ✅ 1,566 files (mostly training images)
- ✅ 1.7 GB saved
- ✅ Will resume from where it stopped
- ✅ No need to re-download existing files

---

## 🎯 Example:

```bash
# If your token is: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/scripts
HF_TOKEN="hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890" python3 download_uvh26.py
```

---

## 💡 Tips:

- Download will automatically **resume** from 1,566 files
- Takes **~2-3 hours** to download remaining ~25GB
- Run in background: Add `&` at end of command
- Token is only used for this download (safe)
