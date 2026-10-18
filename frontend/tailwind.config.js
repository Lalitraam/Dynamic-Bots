/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: '#000000',
        panel: '#0c0c0c',
        gold: '#2F7BFF',
        gray: '#8b8b8b',
      },
      fontFamily: {
        head: ['"Space Grotesk"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"Manrope"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
