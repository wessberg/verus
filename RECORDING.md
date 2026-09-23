# Recording session guide — the four-track booth session

A guide for the next training corpus (BACKLOG B-1, with the acoustic side of the EQ match, B-2): from
walking into the booth until leaving it, plus the short electric session afterwards at home. What
gets recorded and why is in `TRAINING.md` §4; this file is what to *do*.

**The session in one line:** about **3 hours in the booth** for **2 hours of playing** — eight
15-minute takes, alternating picked and fingerstyle (4 × 15 min each), then one ~7-minute take for
the EQ match. Four tracks the whole time: piezo, magnetic, soundhole mic, 12th-fret mic.

**The principles behind everything below:**
- **The model learns whatever the mics hear that the pickups don't — as noise it can't predict.**
  So nothing may reach the mics that doesn't come from the guitar: no talking, humming, click track
  bleed, chair creaks or clothes rustle.
- **The model learns this exact setup.** Mics, chair, guitar position, gains and pickup settings
  stay fixed for the whole session. Moving 5 cm closer to a mic at 20–30 cm changes its level by
  ~2 dB and the stereo image with it.
- **Play what you'll play through the pedal**, across the whole guitar, at every dynamic. The model
  can only be good at what it has heard.
- **Every take is one style** — picked *or* fingerstyle, never both: the style label comes from
  the folder a take is filed in.
- **Every take covers everything**, because the last ~20 % of each take is held out automatically to
  measure the model. End every take with repertoire, the thing that matters most.
- **Don't stop for wrong notes.** The model learns the sound, not the music. Stop only for noise.

---

## The day before

- [ ] **Strings:** the set you normally play, fitted 1–3 days before (settled, not brand-new zing,
      not dead). Don't change them until after the electric EQ-match session.
- [ ] **Disk:** an external drive with **≥ 25 GB free** for the DAW session (the Mac's internal disk
      is too full — `TRAINING.md` §4).
- [ ] **Battery** of the soundhole pickup's preamp (if it's active): fresh.
- [ ] **Picks:** the ones you play. If you use two thicknesses, bring both and note which per take.
- [ ] **The booth sheet** — https://claude.ai/artifact/6FcdUY47gjuzV5oNALoJjj — a tick-through version of this guide with a take
      timer, the current block, and a **Mark noise** button that logs the time into the take. Open
      it on a phone or tablet (or print this file). Tap a take's **Start** at the moment you press
      record, so its noise marks match the recording.

## Entering the booth — setup (~30 min)

1. **Phones in flight mode** (Wi-Fi may stay on for the booth sheet), silent, and **~2 m from the
   guitar** — a phone's cellular radio buzzes in a magnetic pickup.
2. **Quiet clothes**, no watch, bracelet or rings on either hand.
3. **Guitar acclimatizes:** take it out of the case first so it reaches the booth's temperature while
   you set up (tuning stability).
4. **Chair:** one without creaks. Mark its position (tape on the floor). Sit the way you'll sit for
   three hours.
5. **Mics**, placed as you normally do:
   - **mic 1** in front of the soundhole;
   - **mic 2** at the shoulder, pointing down at the 12th fret. Check that your fretting hand and
     the headstock can't hit it when you move up and down the neck.
   Mark the stands' positions on the floor. **Write down** mic models, distances and angles, and
   take photos from the front and the side.
6. **Inputs** — one interface, 96 kHz / 24-bit, tracks named:

   | input | source | notes |
   |---|---|---|
   | 1 | piezo | Hi-Z / instrument input |
   | 2 | magnetic soundhole pickup | Hi-Z / instrument input; any body sensor or mic blend **off** |
   | 3 | mic 1, soundhole | phantom power if needed |
   | 4 | mic 2, 12th fret | phantom power if needed |

   The guitar's own controls (volume, onboard EQ or blend) at your usual setting — **write it
   down**; the same settings are used for the EQ-match session.
7. **Gain:** play the **loudest thing you'll play** — hard strumming, open E and a barre chord high up
   the neck — and set each input to peak around **−10 dBFS**; nothing may ever go above −3 dBFS.
   Then **don't touch any gain again** (the balance between the two mics is part of the stereo image).
8. **Monitoring:** closed headphones at low volume, or none. **No click track in the headphones** —
   it bleeds into the mics but not the pickups. If you need tempo, use a visual metronome.
9. **Room:** fans and air conditioning off, computer outside or silent, door closed. Don't change
   the booth (absorbers, door) during the session — the room is part of the sound.
10. **Test recording (1 min):** play anything, then listen to all four tracks:
    - no hum or buzz on the pickups (move away from lights, dimmers, power supplies);
    - no clipping;
    - both mics sound the way you like — this is the sound the pedal will make;
    - no knocks from hands, picks or the guitar touching a mic.
11. **Room tone:** record **30 s of silence**, guitar in your lap, strings muted, you still.
12. **Tune.**

## Every take, the same way

- **Start:** press record → **5 s of silence** (strings muted) → **three gentle knuckle taps on the
  bridge**, next to the saddle, a second apart → 2 s silence → play.
  (The taps start the piezo and the guitar top at the same instant, so they show the true
  pickup-to-mic delay — an alignment reference. A pick attack doesn't: the mics hear the pick's
  contact ~6.5 ms before the piezo sees the string move.)
- **During:** stay seated, keep the guitar in the same place against your body, don't talk or hum.
  If something noisy happens (a cough, a creak, a knock on a stand), carry on and tap **Mark noise**
  on the booth sheet (or note the time in the log) — it's cut out later. DAW markers usually don't
  survive the WAV export. Stop only if it keeps happening.
- **End:** let the last chord ring out **completely**, then **5 s of silence**, then stop.
- **Write one line in the log** (template at the end).

## The takes

Alternate the styles so string ageing, temperature and fatigue affect both equally:

| # | style | length | then |
|---|---|---|---|
| 1 | picked | 15 min | |
| 2 | fingerstyle | 15 min | 5 min break, tune |
| 3 | picked | 15 min | |
| 4 | fingerstyle | 15 min | 10 min break: tune, water, check disk and levels |
| 5 | picked | 15 min | |
| 6 | fingerstyle | 15 min | 5 min break, tune |
| 7 | picked | 15 min | |
| 8 | fingerstyle | 15 min | 5 min break, tune |
| 9 | picked — **EQ-match take** | ~7 min | takes 10–11 if you have energy; then "Before you leave" |

**Tune before every take**, with recording stopped. Four takes per style = **1 h per style** — the
minimum. If you still have energy, a take 10 (picked) and 11 (fingerstyle) are worth more than
anything else you could do that day: once the mics are moved, a later session can't add to this one.

If you use a **capo** or **other tunings** (drop D, DADGAD …) in real playing, put them in takes 7
and 8 (about half of each), and note it in the log. Standard tuning without a capo should stay
≥ 75 % of the corpus.

### A picked take (≈15 min)

Play the blocks in this order, in your own musical words — the times are rough guides, not a clock:

| block | ~min | what |
|---|---|---|
| Open chords, strummed | 2.5 | progressions (e.g. G–C–D–Em, A–D–E, Am–F–C–G), different strum patterns, soft to loud; let some chords ring out to silence |
| Barre chords and up the neck | 2 | moving shapes up to the 12th fret and beyond; full strums, short chops, palm-muted chugs |
| Single-note lines | 3 | scales and licks on **all six strings**, from open position to the highest frets you use; alternate picking and down-strokes; hammer-ons, pull-offs, slides, small bends, vibrato; some fast, some slow with long ringing notes; **vary where and how hard you pick** (near the bridge / over the soundhole / near the neck, soft to hard) |
| Arpeggios, cross-picking | 2 | picked arpeggios across chords, notes left ringing |
| Dynamics sweep | 1 | one phrase from barely audible to as loud as you play, and back |
| Specials | 1 | natural harmonics (12th, 7th, 5th fret), a few muted scratches, a few single hard pick attacks with silence after each |
| **Repertoire** | 3.5 | songs you'd play through the pedal, picked — including what you'd play on an **electric** through it (single-note lines, riffs) |

### A fingerstyle take (≈15 min)

| block | ~min | what |
|---|---|---|
| Patterns in open position | 2.5 | Travis picking, alternating-thumb bass, your usual patterns |
| Arpeggios | 2 | free strokes and rest strokes, across chord shapes; let notes ring |
| Thumb + melody, up the neck | 2 | bass line with the thumb, melody on top, moving into higher positions |
| Strumming without a pick | 1.5 | thumb, fingers, nails; brush strokes up and down |
| Single notes with the fingers | 1.5 | lines on all strings, legato, slides, vibrato |
| Dynamics sweep + harmonics | 1.5 | one phrase soft → loud → soft; natural and harp/artificial harmonics |
| Percussive (only if you play it) | 0.5 | slaps and body taps as you'd use them — otherwise give the time to repertoire |
| **Repertoire** | 3.5 | fingerstyle pieces you'd play through the pedal |

Nails, flesh or both: whatever you normally do.

### Take 9 — the EQ-match take (picked, ~7 min)

Its magnetic track becomes the acoustic half of the EQ match (B-2); you'll replay exactly this on
the electric, so play **material you can repeat**, all picked, at medium dynamics with some loud
and soft:

| block | ~min | what |
|---|---|---|
| Scale run | 1.5 | one scale up and down **every string**, open position to the 12th fret |
| Chord progression | 2 | one fixed progression, open chords then barre chords, strummed |
| Arpeggios | 1.5 | the same progression, picked note by note |
| A riff or piece | 2 | something you can play the same way on the electric |

It is also a normal picked training take (it gets filed with the others).

## Before you leave

- [ ] **30 s of room tone** again.
- [ ] **Save the DAW session** to the external drive.
- [ ] **Export every take as four mono WAVs**, 96 kHz / 24-bit, **raw**: unprocessed, unaligned,
      untrimmed, from the first silence to the last. Name them
      `takeNN_<style>_piezo.wav`, `_magnetic.wav`, `_mic_soundhole.wav`, `_mic_12th.wav`
      (style = `pick` or `finger`), plus `roomtone_start/end_*.wav`.
      The alignment, the stereo target files and the `input/` layout are done afterwards — see
      "Preparing the files" below. **Don't align, trim or process anything in the DAW.**
- [ ] **Copy the setup notes, the photos and the log** next to the exports.
- [ ] Open one or two exported files to check they play.
- [ ] **Leave the mics standing** if you might add takes another day — a later session only fits
      this corpus with the identical setup.

## Afterwards at home — the electric side of the EQ match (~20 min)

- Same interface, **the magnetic pickup's input (2), at the same gain**, electric straight in
  (no pedals, no amp sim).
- **Neck humbucker**, volume and tone where you normally play them (note them).
- **Play along with take 9's recording in headphones** — the same material, in the same order, same
  pick, same dynamics. The closer the two performances, the more exact the EQ: in round 07's test an
  identical performance matched within ~0.2 dB and an unrelated 5-minute passage ~1.5 dB; playing
  along lands in between.
- 5 s of silence at the start and end. Save as `electric_humbucker.wav`.
- Bonus: the same again on the **bridge humbucker** → `electric_humbucker_bridge.wav`.
- Also check the input impedance question while you're at it (B-3).

---

## Preparing the files (after the session)

The prep step (BACKLOG B-23) will do all of this from the raw exports; this section is what it does
and why — and the procedure if you do it by hand in the DAW. The rule behind every step: **whatever
you do to one track, do the same to all four, except the steps named here** — the model learns the
exact relationship between the pickups and the mics, so any processing that differs between tracks
becomes part of what it learns.

### 1. Keep the takes whole — split only around noise

- **One file per track per take**, from the first note to the end of the ring-out. Don't split takes
  into songs or blocks: the last 20 % of every *file* becomes validation data, so every extra file
  adds another held-out tail and changes what the model is measured on.
- **Noise from the log** (a cough, a creak, a knock on a stand): cut from 1 s before to 2 s after it
  and **split the take there** into two files (`take03a`, `take03b`) — never splice the two parts
  back together; the joint would be a jump the mics never heard. Drop a piece shorter than ~30 s.
- **At every cut, a 5 ms fade** (raised-cosine) in or out, on all four tracks identically.
- **Trim the start** after aligning: remove the silence and the three taps, keeping ~1 s before the
  first note. **Keep the end**: the full ring-out plus ~2 s. Short silences inside a take are fine —
  the model should learn that silence stays silence.

### 2. Alignment — one shared delay for the mic pair

- **What gets aligned:** the pickups are the reference and stay where they are (same interface, same
  clock, no acoustic path). The two mics move **together, by the same number of samples**, so that
  the *earlier* mic lines up with the piezo. That removes the ~0.6–1.5 ms the sound needs to travel
  to the mics, so the pedal doesn't add that delay — the same convention as the old corpus.
- **How:** on the **three bridge taps** at the start of the take: find the first rise of each tap in
  the piezo and in the earlier mic, and take the median of the three offsets. Whole samples at
  96 kHz (10 µs) are precise enough.
- **Check:** after aligning, onsets of notes that follow a silence should line up in the midband
  (80 Hz–1 kHz) with a median near 0 ms — the old corpus measures 0.0 ms fingerstyle and −1.5 ms
  picked (round 08). In the high band the mics *lead* by ~6.5 ms: that's the pick or nail contact,
  which the piezo can't see (round 06) — **don't align to that**, and don't align to pick attacks.
- **Why not align by cross-correlation of the playing:** on this guitar it's ambiguous. A strong
  body resonance near 175 Hz makes the low-band correlation show two almost equal peaks, 2.9 ms apart
  with opposite signs, and which one wins changes along the take; broadband methods are pulled
  toward the high-band lead instead (both measured on the old corpus, round 08).
- **Keep the offset between the two mics exactly as recorded** — never align them to each other.
  Their relative delay *is* the stereo image. Note it down (the prep step reports it): the pedal's
  mono option can use it for a delay-aligned sum (B-22).
- **Consistency:** the offset should be the same within a few samples in every take. A jump of more
  than ~10 samples (0.1 ms ≈ 3.5 cm) means you, a mic or the chair moved — worth knowing, not fatal.

### 3. Polarity — check it, don't "phase-correct"

- **Polarity:** after aligning, the piezo and the soundhole mic should correlate *positively* in the
  low band at zero offset, as in the old corpus (+0.4 to +0.5). If it's negative (a preamp or cable
  inverts somewhere), **invert both mics together** — never one of them: their relative polarity is
  part of the sound.
- **No phase alignment plugins** (Auto-Align and similar), no all-pass or phase-rotation, no
  per-mic nudging. If you phase-align the two mics when you mix, that is a choice for the *output*
  (the pedal's mono/stereo options, B-22), not something to bake into what the model learns.

### 4. Levels — no normalization

- **Don't normalize — not per take, not per track, not by RMS or peak.** Two reasons:
  - The **balance between the two mics** is part of the stereo image; normalizing each separately
    changes it take by take.
  - The **level of a take is information**: a soft take and a loud take differ in timbre, and the
    model learns that from their levels relative to each other. (Absolute level doesn't matter —
    training applies a random gain, the same to input and target, of −24 to +12 dB.)
- **One constant per track, for the whole session**, if any: the prep step brings the pickups to the
  old corpus's playing level (piezo RMS about −28 to −29 dBFS) and the mic pair — with **one shared
  gain** — to where the soundhole mic sat against the piezo in the old corpus (mic minus piezo RMS
  +1.8 dB picked, −2.4 dB fingerstyle). That puts the new data where the warm-started model
  expects it.
- **Headroom:** after that gain, no peak may reach full scale; 24-bit PCM. (The old corpus's mic
  peaked at −1.7 dBFS — too close; −10 dBFS at recording is the safer target.)
- **No dither**, no bit-depth or sample-rate change: 96 kHz / 24-bit in, 96 kHz / 24-bit out (the
  pipeline makes its own 48 kHz versions for the realtime model).

### 5. No other processing

- **No EQ, compression, limiting, de-noising, de-essing, reverb, or fades other than at cuts.** The
  mics' sound as recorded *is* the target. De-noising in particular leaves artifacts the model would
  learn to reproduce.
- **DC offset:** removed only if a track actually has one (the prep step measures it).
- **Rumble:** if the mics show energy below ~40 Hz that isn't the guitar (stands, floor, air
  handling), a zero-phase 30 Hz high-pass on **both mics identically** — only if measured. The lowest
  string is at 82 Hz (73 Hz in drop D).

### 6. Files and checks

- **Stereo target** per take: channel 1 = soundhole mic, channel 2 = 12th-fret mic; the pickups as
  mono sources; the magnetic take's target a **hard link** to the piezo take's (layout: `TRAINING.md`
  §4 "The four-track session").
- **EQ match:** take 9's magnetic track also goes to `input/eqmatch/acoustic_magnetic.wav` —
  there, **keep the silence at start and end** (it measures the noise floor) and remove only the taps.
- **Checks per take** (the prep step prints them): no clipping on any track, pickups free of hum,
  noise floor against the room tone, the mic offset and polarity, levels against the other takes,
  style and length.

## Log sheet

| take | style | start | length | pick / nails | capo / tuning | notes (noise at mm:ss, anything odd) |
|---|---|---|---|---|---|---|
| 1 | pick | | | | | |
| 2 | finger | | | | | |
| 3 | pick | | | | | |
| 4 | finger | | | | | |
| 5 | pick | | | | | |
| 6 | finger | | | | | |
| 7 | pick | | | | | |
| 8 | finger | | | | | |
| 9 | pick (EQ match) | | | | | |

Setup notes: mic 1 ______ at ___ cm, angle ___ · mic 2 ______ at ___ cm, angle ___ ·
guitar controls ______ · gains ch1 ___ ch2 ___ ch3 ___ ch4 ___ · strings fitted on ______
