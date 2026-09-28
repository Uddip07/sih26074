# Demo Video

The demo video is optional, but recommended.

## Demo video link

`<PASTE_YOUTUBE_OR_GOOGLE_DRIVE_VIDEO_LINK_HERE>`

## Suggested demo flow (about 3 minutes)

1. **Problem:** one block forecast covers ~100 villages from the Ghats to the rain-shadow.
2. **Dashboard:** block forecast (left) vs downscaled Gram Panchayat forecast (right); move the day slider
   (Today → Day 7).
3. **A panchayat:** search or "My panchayat" → 8-day block-vs-panchayat chart, likely range, "why it differs".
4. **Advisories:** warning level, crop advisories, switch मराठी / हिन्दी / English, download the PDF bulletin,
   show the SMS.
5. **Scorecard:** accuracy on never-seen panchayats vs copying the block value.
6. **Officer:** review/approve, publish, upload an official block-forecast CSV.

Start the app with `uvicorn src.main:app --port 8080`. Direct links for recording, e.g.
`http://127.0.0.1:8080/?gp=185267&day=2` (a panchayat) or `?tab=score` (scorecard).
