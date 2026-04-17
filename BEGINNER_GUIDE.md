# 🎓 Beginner's Guide: Deploy to Hugging Face Spaces

## Step-by-Step Guide for Complete Beginners

This guide assumes **zero prior experience** with Hugging Face Spaces or Git deployment. We'll walk through **every single step** with explanations.

---

## 📋 Before You Start

### 1. Install Git LFS (Required!)

**Windows:**
1. Go to https://git-lfs.github.com/
2. Click "Download"
3. Run the installer
4. Open **Command Prompt** (Windows Key + R, type `cmd`, press Enter)
5. Type: `git lfs install` and press Enter
6. You should see: "Git LFS initialized."

**Mac:**
1. Open **Terminal** (search for "Terminal" in Spotlight)
2. Type: `brew install git-lfs` and press Enter
3. If you don't have Homebrew, install it first: https://brew.sh/
4. Type: `git lfs install` and press Enter

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
```

### 2. Verify Git LFS is Working

In Command Prompt or Terminal:
```bash
git lfs version
```

You should see something like: `git-lfs/3.4.0`

✅ **If you see a version number, you're ready!**

### 3. Get API Keys (Free!)

You need at least **one** of these (I recommend Google Gemini):

**Option 1: Google Gemini API Key (Recommended)**
1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Click "Create API key in new project"
5. **Copy the key** that appears (starts with `AIza...`)
6. Save it in a notepad - you'll need it later!

**Option 2: Groq API Key (Alternative)**
1. Go to https://console.groq.com
2. Sign up for a free account
3. Go to "API Keys" section
4. Click "Create API Key"
5. **Copy the key** (starts with `gsk_...`)
6. Save it!

**Option 3: Hugging Face Token (Optional - for better images)**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name: "MAdVerse"
4. Select "Read" permission
5. Click "Generate"
6. **Copy the token** (starts with `hf_...`)
7. Save it!

---

## 🚀 Part 1: Create Your Hugging Face Space

### Step 1: Create Hugging Face Account (if you don't have one)

1. Go to https://huggingface.co/join
2. Fill in:
   - Email address
   - Password
   - Username (e.g., `yourname`)
3. Click "Sign Up"
4. Check your email and verify your account
5. Complete your profile

### Step 2: Create a New Space

1. **Go to**: https://huggingface.co/new-space
   
2. **Fill in the form** (see screenshot below):

   ```
   ┌─────────────────────────────────────────┐
   │ Create a new Space                      │
   ├─────────────────────────────────────────┤
   │                                         │
   │ Owner: [your-username]                  │
   │                                         │
   │ Space name: [madverse]                  │
   │ ▢ Make this Space private              │
   │                                         │
   │ License: MIT                            │
   │                                         │
   │ Select the Space SDK:                   │
   │ ○ Gradio  ○ Streamlit  ● Docker        │
   │ ○ Static                                │
   │                                         │
   │ Space hardware:                         │
   │ ● CPU basic - Free (2 vCPU, 16GB RAM)  │
   │ ○ CPU upgrade - $0.03/hour             │
   │ ○ T4 small - $0.60/hour                │
   │                                         │
   │ Space template:                         │
   │ ● Blank                                 │
   │                                         │
   │        [Create Space]                   │
   └─────────────────────────────────────────┘
   ```

3. **Enter these exact values**:
   - **Owner**: Your Hugging Face username (auto-filled)
   - **Space name**: `madverse` (or any name you like)
   - **License**: Select "MIT"
   - **SDK**: Click on **"Docker"** (important!)
   - **Hardware**: Select **"CPU basic - Free"** (the free option)
   - **Template**: Keep "Blank" selected

4. **Click the big "Create Space" button** at the bottom

5. **Wait 2-3 seconds** - you'll be redirected to your new Space page

---

## 🎯 Part 2: After Clicking "Create Space"

### What You'll See

You'll land on a page that looks like this:

```
┌────────────────────────────────────────────────────┐
│ your-username / madverse                           │
│ [No app file]                                      │
│                                                    │
│ This Space has no app yet. Clone the repo and     │
│ add your files to get started.                    │
│                                                    │
│ Clone this repository:                            │
│ git clone https://huggingface.co/spaces/          │
│           your-username/madverse                   │
└────────────────────────────────────────────────────┘
```

**Don't worry!** This is normal. Now we'll add your app files.

---

## 📦 Part 3: Choose Your Deployment Method

### Method A: Automatic Script (Easiest!) ⭐

This is the **easiest way** - the script does everything for you!

#### For Windows Users:

1. **Open Command Prompt**:
   - Press `Windows Key + R`
   - Type `cmd`
   - Press Enter

2. **Navigate to your project folder**:
   ```cmd
   cd P:\KHUB\product-labs\MAdVerse
   ```
   (Replace with your actual path)

3. **Run the deployment script**:
   ```cmd
   scripts\deploy-to-hf.bat https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```
   
   **Replace `YOUR_USERNAME` with your actual Hugging Face username!**
   
   Example:
   ```cmd
   scripts\deploy-to-hf.bat https://huggingface.co/spaces/johnsmith/madverse
   ```

4. **What the script does** (you'll see this output):
   ```
   ================================================
     MAdVerse - Hugging Face Spaces Deployment
   ================================================
   
   [OK] Git LFS is installed
   [OK] .gitattributes found
   [OK] FAISS index OK (345 MB)
   
   ================================================
     Deployment Target: https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ================================================
   
   Deploy to this Space? (y/n):
   ```

5. **Type `y` and press Enter**

6. **Wait** (5-10 minutes) while it:
   - Clones your Space
   - Copies all files
   - Sets up Git LFS
   - Uploads everything to Hugging Face

7. **When done**, you'll see:
   ```
   ================================================
     Deployment Complete!
   ================================================
   
   Next steps:
   
   1. Add API keys in Space Settings:
      https://huggingface.co/spaces/YOUR_USERNAME/madverse/settings
   ```

8. **Jump to Part 4 below** to add your API keys!

#### For Mac/Linux Users:

1. **Open Terminal**

2. **Navigate to project**:
   ```bash
   cd /path/to/MAdVerse
   ```

3. **Make script executable**:
   ```bash
   chmod +x scripts/deploy-to-hf.sh
   ```

4. **Run the script**:
   ```bash
   ./scripts/deploy-to-hf.sh https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```
   (Replace `YOUR_USERNAME` with your Hugging Face username)

5. **Type `y` when asked**

6. **Wait** while it deploys (5-10 minutes)

7. **Jump to Part 4 to add API keys!**

---

### Method B: Manual Steps (If Script Doesn't Work)

If the automatic script doesn't work, follow these manual steps:

#### Step 1: Download Git Files from Hugging Face

1. **Copy your Space URL**. It looks like:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```

2. **Open Command Prompt** (Windows) or **Terminal** (Mac/Linux)

3. **Choose a folder** where you want to work:
   ```cmd
   cd Desktop
   ```
   (This will create a folder on your Desktop)

4. **Clone the Space**:
   ```cmd
   git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```
   
   **Replace `YOUR_USERNAME`** with your actual username!
   
   Example:
   ```cmd
   git clone https://huggingface.co/spaces/johnsmith/madverse
   ```

5. **Wait** - you'll see:
   ```
   Cloning into 'madverse'...
   remote: Enumerating objects: 4, done.
   remote: Counting objects: 100% (4/4), done.
   remote: Compressing objects: 100% (3/3), done.
   Receiving objects: 100% (4/4), done.
   ```

6. **Enter the folder**:
   ```cmd
   cd madverse
   ```

#### Step 2: Copy Your Project Files

**Windows:**

1. **Open two File Explorer windows**:
   - Window 1: Your original MAdVerse project (`P:\KHUB\product-labs\MAdVerse`)
   - Window 2: Your new Space folder (`Desktop\madverse`)

2. **Copy these folders** from Window 1 to Window 2:
   - `app` folder
   - `scripts` folder
   - `embeddings` folder
   - `processed` folder
   - `data\annotations` folder (NOT `data\images`!)
   - `data\indices` folder

3. **Copy these files** from Window 1 to Window 2:
   - All `.py` files (`run.py`, `DatasetLoad.py`, etc.)
   - `Dockerfile`
   - `requirements.txt`
   - `.gitattributes`
   - `.gitignore`
   - `.dockerignore`
   - `.env.example`
   - `docker-compose.yml`
   - `.spacesconfig.yml`

4. **Special: Rename README**:
   - Find `README_HF_SPACE.md` in Window 1
   - Copy it to Window 2
   - In Window 2, rename the existing `README.md` to `README_old.md`
   - Rename `README_HF_SPACE.md` to `README.md`

**DO NOT copy**:
- ❌ `.git` folder
- ❌ `.venv` folder
- ❌ `data\images` folder (it's 12 GB!)
- ❌ `outputs` folder
- ❌ `uploads` folder
- ❌ `products_db` folder

#### Step 3: Setup Git LFS in Your Space

1. **Still in Command Prompt/Terminal**, in the `madverse` folder:

2. **Initialize Git LFS**:
   ```cmd
   git lfs install
   ```

3. **You should see**:
   ```
   Updated Git hooks.
   Git LFS initialized.
   ```

#### Step 4: Add All Files to Git

1. **Add all files**:
   ```cmd
   git add .
   ```

2. **Check what will be uploaded**:
   ```cmd
   git status
   ```

3. **You should see** a long list of files in green

#### Step 5: Commit Your Files

1. **Create a commit**:
   ```cmd
   git commit -m "Deploy MAdVerse AI to Hugging Face Spaces"
   ```

2. **You should see**:
   ```
   [main abc1234] Deploy MAdVerse AI to Hugging Face Spaces
    100 files changed, 5000 insertions(+)
    create mode 100644 app/main.py
    create mode 100644 Dockerfile
    ...
   ```

#### Step 6: Push to Hugging Face

1. **Upload to Hugging Face**:
   ```cmd
   git push origin main
   ```

2. **Enter your Hugging Face credentials** if asked:
   - Username: Your HF username
   - Password: Your HF password (or access token)

3. **Wait** (5-10 minutes) - you'll see:
   ```
   Uploading LFS objects: 100% (5/5), 410 MB | 2 MB/s, done.
   Enumerating objects: 150, done.
   Counting objects: 100% (150/150), done.
   Writing objects: 100% (150/150), 25 KB | 2 MB/s, done.
   ```

4. **When done**, you'll see:
   ```
   To https://huggingface.co/spaces/YOUR_USERNAME/madverse
      abc1234..def5678  main -> main
   ```

✅ **Upload complete!**

---

## 🔑 Part 4: Add Your API Keys (IMPORTANT!)

After uploading files, you need to add your API keys so the app can work.

### Step 1: Go to Space Settings

1. **Open your Space page**:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```

2. **Click the "Settings" tab** at the top (next to "Files", "Community")

3. **Scroll down** to the section called **"Repository secrets"**

### Step 2: Add Your First API Key

1. **In the "Repository secrets" section**, you'll see:
   ```
   ┌──────────────────────────────────────┐
   │ Repository secrets                   │
   ├──────────────────────────────────────┤
   │ Secret keys are private. They won't │
   │ be visible in the Space logs.       │
   │                                      │
   │ [+ New secret]                       │
   └──────────────────────────────────────┘
   ```

2. **Click "[+ New secret]"**

3. **You'll see two fields**:
   ```
   Name:  [________________]
   Value: [________________]
   ```

4. **Add your Google Gemini key**:
   - **Name**: Type exactly: `GOOGLE_API_KEY`
   - **Value**: Paste your API key (the one starting with `AIza...`)

5. **Click "Add secret"**

6. **You should see**:
   ```
   ✓ Secret GOOGLE_API_KEY added
   ```

### Step 3: Add More Keys (Optional but Recommended)

**Repeat Step 2** for each additional key you have:

**For Hugging Face Token** (better images):
- **Name**: `HF_TOKEN`
- **Value**: Your HF token (starts with `hf_...`)

**For Groq** (alternative text generation):
- **Name**: `GROQ_API_KEY`
- **Value**: Your Groq key (starts with `gsk_...`)

### Step 4: Verify Keys Are Added

You should now see a list:
```
┌──────────────────────────────────────┐
│ Repository secrets                   │
├──────────────────────────────────────┤
│ • GOOGLE_API_KEY     [Edit] [Delete] │
│ • HF_TOKEN          [Edit] [Delete] │
└──────────────────────────────────────┘
```

✅ **API keys configured!**

---

## ⏳ Part 5: Wait for Build to Complete

### What Happens Now

After you push files and add API keys, Hugging Face will:

1. **Build your Docker container** (~10-15 minutes)
2. **Start your application** (~1-2 minutes)
3. **Run health checks** (~30 seconds)

### How to Monitor Progress

1. **Go to your Space page**:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/madverse
   ```

2. **Click the "Logs" tab** (at the top)

3. **You'll see real-time build output**:
   ```
   Building Docker image...
   Step 1/15 : FROM python:3.11-slim
   Step 2/15 : ENV PYTHONDONTWRITEBYTECODE=1
   Step 3/15 : RUN apt-get update && apt-get install...
   ...
   Successfully built abc123def456
   Starting container...
   
   ============================================================
     MAdVerse AI — Container Startup
   ============================================================
   [STARTUP] Loading environment from /app/.env
   [STARTUP] HuggingFace cache directory: /app/.cache/huggingface
   [CHECK] Validating FAISS index...
   ✓ FAISS index OK (345 MB)
   ✓ Metadata file OK
   
   ============================================================
     Starting MAdVerse AI Server
     Port: 7860
   ============================================================
   
   INFO:     Started server process [1]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:7860
   ```

4. **Look for these success messages**:
   - ✅ `✓ FAISS index OK`
   - ✅ `Application startup complete`
   - ✅ `Uvicorn running on http://0.0.0.0:7860`

### Expected Timeline

| Time | What's Happening |
|------|------------------|
| **0-2 min** | Downloading your code |
| **2-5 min** | Installing system dependencies |
| **5-10 min** | Installing PyTorch and Python packages |
| **10-12 min** | Building Docker image |
| **12-13 min** | Starting container |
| **13-14 min** | Loading FAISS index |
| **14-15 min** | Starting FastAPI server |
| **15 min** | ✅ **READY!** |

### How to Know It's Ready

1. **At the top of your Space page**, you'll see a status badge:
   ```
   Before: [Building...]
   After:  [Running ✓]
   ```

2. **The "App" tab** will show your application instead of logs

3. **You can click "App"** and see the MAdVerse interface!

---

## 🎉 Part 6: Test Your Deployed App

### Step 1: Access Your Space

**Your app is now live at**:
```
https://YOUR_USERNAME-madverse.hf.space
```

Example:
```
https://johnsmith-madverse.hf.space
```

### Step 2: Test the App

1. **Open the URL** in your browser

2. **You should see** the MAdVerse home page with:
   - Title: "MAdVerse - AI Ad Generation"
   - A text box for product description
   - Language selector
   - "Generate Ad" button

3. **Try generating an ad**:
   - In the text box, type:
     ```
     Wireless headphones with noise cancellation, $149, perfect for commuters
     ```
   - Select a language (e.g., "English")
   - Click "Generate Ad"

4. **Wait 10-15 seconds**

5. **You should see**:
   - ✅ Generated product image
   - ✅ Product title and description
   - ✅ Instagram caption
   - ✅ WhatsApp message
   - ✅ Hashtags
   - ✅ Multi-language versions

### Step 3: Test Other Features

**Dataset Explorer**:
```
https://YOUR_USERNAME-madverse.hf.space/dataset
```
- Should show statistics about 50,000+ ads
- Categories, languages, dataset breakdown

**API Documentation**:
```
https://YOUR_USERNAME-madverse.hf.space/docs
```
- Interactive API documentation
- Try out API endpoints

**Health Check**:
```
https://YOUR_USERNAME-madverse.hf.space/api/health
```
- Should return JSON:
  ```json
  {
    "status": "healthy",
    "faiss_status": "ready",
    "embeddings_count": 50000
  }
  ```

---

## ✅ Success Checklist

Your deployment is successful if:

- [ ] Space status shows "Running" (not "Error" or "Building")
- [ ] App URL loads without errors
- [ ] Can generate a test ad
- [ ] Generated image appears (not just a placeholder)
- [ ] Multi-language captions are generated
- [ ] Dataset explorer loads
- [ ] API docs accessible at `/docs`
- [ ] Health check returns `{"status": "healthy"}`

**If all checked ✅ - Congratulations! Your app is live! 🎉**

---

## 🆘 Troubleshooting for Beginners

### Problem: "Building..." never finishes

**Check**:
1. Go to "Logs" tab
2. Look for error messages (in red)

**Common causes**:
- **"FAISS index is a Git LFS pointer"**
  - **Fix**: You didn't run `git lfs pull` before deploying
  - **Solution**: In your local project folder, run:
    ```cmd
    git lfs pull
    python scripts/check_faiss.py
    ```
  - Then re-deploy (push again)

### Problem: "Application failed to start"

**Check logs for**:
- **"ModuleNotFoundError"**
  - **Fix**: Missing dependency
  - **Solution**: Check `requirements.txt` has all packages

- **"Port already in use"**
  - **Fix**: Wrong port configuration
  - **Solution**: Check `Dockerfile` exposes port 7860

### Problem: Generated images are plain colors

**This is normal if**:
- You didn't add `HF_TOKEN` in secrets

**Fix**:
1. Go to Settings → Repository secrets
2. Add `HF_TOKEN` with your Hugging Face token
3. Space will restart automatically
4. Try generating again

### Problem: "No text generation API keys"

**Check logs** - you'll see:
```
⚠  WARNING: No text generation API keys found
```

**This means**:
- You forgot to add `GOOGLE_API_KEY` or `GROQ_API_KEY`
- App will work but use slower free alternative

**Fix**:
1. Go to Settings → Repository secrets
2. Add `GOOGLE_API_KEY` with your Gemini key
3. Space restarts
4. Quality improves!

### Problem: "I don't see my API keys in logs"

**This is CORRECT!**
- API keys are secret and won't show in logs
- They're injected as environment variables
- You should see: `✓ Text generation API key(s) configured`

### Need More Help?

1. **Check the full guide**: `HUGGINGFACE_DEPLOYMENT.md`
2. **Ask on Discord**: https://hf.co/join/discord
3. **Post on forums**: https://discuss.huggingface.co/
4. **Check HF Docs**: https://huggingface.co/docs/hub/spaces

---

## 📝 Summary for Absolute Beginners

### What You Did

1. ✅ Installed Git LFS
2. ✅ Got API keys (Gemini/Groq)
3. ✅ Created a Hugging Face Space
4. ✅ Deployed your MAdVerse app
5. ✅ Added API keys as secrets
6. ✅ Waited for build to complete
7. ✅ Tested your live app

### What You Have Now

- 🌐 **Live web app** at `https://YOUR_USERNAME-madverse.hf.space`
- 🆓 **Completely free** (on cpu-basic tier)
- 🚀 **Accessible worldwide** (anyone can use it)
- 📊 **50,000+ ads** searchable with AI
- 🎨 **AI-powered ad generation**
- 🌍 **50+ languages** supported

### Total Time Spent

- Setup: ~10 minutes
- Deployment: ~5 minutes
- Build wait: ~15 minutes
- **Total: ~30 minutes**

### Total Cost

**$0.00** (FREE forever on Hugging Face Spaces!) ✨

---

## 🎊 You Did It!

**You've successfully deployed an AI application to the cloud!**

This is a real achievement - you've:
- ✅ Worked with Git and version control
- ✅ Used Docker containers
- ✅ Deployed to cloud infrastructure
- ✅ Configured environment variables
- ✅ Set up API integrations

**Share your Space**:
```
My MAdVerse AI App:
https://YOUR_USERNAME-madverse.hf.space

Generate professional ads with AI! 🎨
```

---

**Congratulations! 🎉🚀**

**Questions?** Re-read any section above or check `HUGGINGFACE_DEPLOYMENT.md` for more details.
