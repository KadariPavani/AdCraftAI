# 🚀 MAdVerse Deployment Guide
## Get Your Permanent Public Endpoint

Your project is ready to deploy! Here are **3 options** to get a public URL for your mentors.

---

## ✅ **OPTION 1: Render.com (RECOMMENDED - Free Tier Available)**

Your project already has `render.yaml` configured!

### Steps:

1. **Push to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Sign up on Render**
   - Go to: https://render.com
   - Sign up with GitHub account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `product-labs/MAdVerse`
   - Render will auto-detect the `render.yaml` file

4. **Set Environment Variables** (in Render Dashboard)
   Required (at least one):
   - `GOOGLE_API_KEY` - Get from https://aistudio.google.com/apikey
   - `GROQ_API_KEY` - Get from https://console.groq.com
   
   Optional:
   - `HF_TOKEN` - For image generation
   - `TOGETHER_API_KEY` - Alternative image generation

5. **Deploy!**
   - Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - **Your endpoint will be:** `https://madverse-adcraft.onrender.com`

### Your API Endpoints:
- **Main**: `https://madverse-adcraft.onrender.com/api/generate`
- **Docs**: `https://madverse-adcraft.onrender.com/docs`
- **Health**: `https://madverse-adcraft.onrender.com/api/health`

**Note:** Free tier spins down after 15 min of inactivity. First request after sleep takes ~30 seconds.

---

## ✅ **OPTION 2: Hugging Face Spaces (Free)**

Your Dockerfile is optimized for HF Spaces!

### Steps:

1. **Create Space**
   - Go to: https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `madverse-adcraft`
   - Select: Docker
   - License: MIT

2. **Upload Your Code**
   ```bash
   # Clone your HF Space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/madverse-adcraft
   cd madverse-adcraft
   
   # Copy your project files
   cp -r P:\KHUB\product-labs\MAdVerse/* .
   
   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

3. **Set Environment Variables** (in HF Space Settings)
   - Go to Settings → Variables
   - Add: `GOOGLE_API_KEY`, `GROQ_API_KEY`, etc.

4. **Your endpoint will be:**
   `https://YOUR_USERNAME-madverse-adcraft.hf.space`

---

## ✅ **OPTION 3: ngrok (Quick & Temporary - For Testing)**

Best for immediate testing, but URL changes on restart.

### Steps:

1. **Download ngrok**
   - Go to: https://ngrok.com/download
   - Sign up for free account
   - Download Windows version

2. **Install & Authenticate**
   ```bash
   # Extract ngrok.exe to your project folder
   ngrok config add-authtoken YOUR_AUTHTOKEN
   ```

3. **Start Your Local Server**
   ```bash
   python run.py
   ```

4. **In Another Terminal, Run ngrok**
   ```bash
   ngrok http 8000
   ```

5. **Get Your Public URL**
   - ngrok will show: `Forwarding https://abc123.ngrok.io -> localhost:8000`
   - **Your endpoint:** `https://abc123.ngrok.io/api/generate`

**Note:** URL changes every time you restart ngrok. Good for testing only.

---

## 📋 **Complete API Endpoint List for Your Mentors**

Once deployed, share these endpoints:

### **Ad Generation**
- `POST /api/generate` - Main ad generation endpoint
- `POST /api/products/{product_id}/generate` - Generate for specific product

### **Product Management**
- `POST /api/products` - Create product
- `GET /api/products` - List all products
- `GET /api/products/{product_id}` - Get specific product

### **AI Features**
- `POST /api/parse-prompt` - Parse natural language
- `POST /api/enhance` - Enhance descriptions
- `POST /api/describe` - Describe images

### **System Info**
- `GET /api/health` - Health check
- `GET /api/stats` - System statistics
- `GET /api/docs` - Interactive API documentation

---

## 🔧 **Deployment Checklist**

- [ ] Environment variables configured (API keys)
- [ ] Repository pushed to GitHub
- [ ] Deployment service chosen (Render/HF/ngrok)
- [ ] Service deployed and running
- [ ] Health check endpoint tested: `/api/health`
- [ ] API documentation accessible: `/docs`
- [ ] Endpoint URL shared with mentors

---

## 📞 **Share With Your Mentors**

**Your Public Endpoint:**
```
https://YOUR_DEPLOYED_URL.com
```

**Main API Endpoint:**
```
POST https://YOUR_DEPLOYED_URL.com/api/generate
```

**API Documentation:**
```
https://YOUR_DEPLOYED_URL.com/docs
```

**Example Request:**
```bash
curl -X POST "https://YOUR_DEPLOYED_URL.com/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Red Bull energy drink for sports enthusiasts",
    "language": "en",
    "num_variations": 2
  }'
```

---

## ⚠️ **Important Notes**

1. **API Keys**: Keep them secret! Never commit `.env` file to git.
2. **Free Tier Limits**: 
   - Render: Spins down after 15 min inactivity
   - HF Spaces: 2 CPU cores, 16GB RAM max
3. **First Request**: May take 30-60 seconds (cold start)
4. **Persistent Storage**: Render provides 1GB disk for outputs

---

## 🆘 **Need Help?**

- Render Docs: https://render.com/docs
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- ngrok Docs: https://ngrok.com/docs

---

**Recommended: Use Render.com for production deployment** ✨
