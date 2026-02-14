# 💘 Overengineered Valentine

A playful, over-the-top Valentine’s Day web experience built as a single-file application.

It includes animated floating hearts, romantic background music, AI-generated poetry, secret message reveal, password protection, mobile responsiveness, and high‑resolution card export.

---

## ✨ Features

* 💘 Canvas-based floating heart particle system
* 🎵 Romantic background music using Web Audio API
* 🤖 AI-generated poem (serverless-ready endpoint)
* 💌 Click-to-reveal secret message
* 🔐 Optional password protection (client-side gate)
* 📱 Fully responsive layout (mobile + desktop)
* 📥 Download card as high-resolution PNG
* 🔗 Deployable via GitHub Pages (static version)

---

## 📂 Project Structure

```
.
├── index.html        # Main single-file application
├── README.md         # Project documentation
└── requirements.txt  # Backend dependencies (optional AI server)
```

---

## 🚀 Quick Start (Static Version)

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/valentine-project.git
   cd valentine-project
   ```

2. Open `index.html` directly in your browser.

No build step required.

---

## 🤖 Enabling AI Poem Generation

The frontend expects a POST endpoint:

```
/api/generate-poem
```

You can deploy this as a serverless function (Vercel, Netlify, etc.).

### Example Node.js Serverless Function

```javascript
import OpenAI from "openai";

export default async function handler(req, res) {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Write a romantic Valentine's Day poem." }
    ],
  });

  res.status(200).json({ poem: completion.choices[0].message.content });
}
```

Set your environment variable:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 🌍 Deploy to GitHub Pages

1. Push your repository to GitHub.
2. Go to **Settings → Pages**.
3. Select branch: `main` and folder: `/root`.
4. Save.

Your site will be available at:

```
https://YOUR_USERNAME.github.io/valentine-project/
```

Note: AI features require a deployed backend.

---

## 🔐 Security Note

Password protection in the frontend is for playful gating only. It is not secure for protecting sensitive information.

---

## 📜 License

MIT — free to fork, remix, and share the love.

---

Built with ❤️ for Valentine’s Day.
