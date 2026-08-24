/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FFFCF8",
        card: "#FFFFFF",
        ink: "#1A1A1A",
        muted: "#6B6B6B",
        line: "#E8E0D8",
        accent: "#0F766E",
        accent2: "#E85D3F",
        sand: "#F5EFE6",
      },
      fontFamily: {
        sans: ["Inter", "Satoshi", "system-ui", "sans-serif"],
        serif: ["Newsreader", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
