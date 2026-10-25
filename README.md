# Shorts pe RTX 4090 — RunPod Serverless

Worker personalizat pentru LTX-2 original (19B), nu LTX-2.5. Primește promptul tău și exportă un MP4 cu video și audio. Nu publică automat pe YouTube.

**Stare:** sursă pregătită, verificări locale de sintaxă și export media. Imaginea Docker și inferența pe RTX 4090 NU au fost încă testate. Memoria suficientă, timpul, calitatea și absența vocilor trebuie verificate printr-o generare reală înainte de utilizare repetată. CFG=1 din modelul distilled nu aplică separarea obișnuită a negative promptului; instrucțiunile audio sunt incluse și în promptul pozitiv. Nu există garanție automată de absență a vocilor.

## 1. Creează repository-ul GitHub

1. Deschide https://github.com/new și creează un repository numit `runpod-shorts-4090`, pe contul tău personal. Varianta Public simplifică distribuirea imaginii; nu include tokenuri sau prompturi private în cod.
2. Extrage arhiva primită de la Codex. În repository: **uploading an existing file** / **Add file → Upload files**.
3. Încarcă **conținutul** folderului, nu arhiva ZIP și nu folderul părinte. `Dockerfile`, `handler.py` și `workflow.py` trebuie să fie în rădăcina repository-ului.
4. Include și `.github/workflows/build.yml`. Pe macOS, `Command + Shift + .` afișează fișierele ascunse în Finder. Alternativ, după încărcarea celorlalte fișiere, **Add file → Create new file**, numele `.github/workflows/build.yml`, apoi copiază conținutul din fișierul local corespunzător.
5. Commit pe `main`. La **Actions → Build RunPod worker**, urmărește rezultatul. Build-ul descarcă multe dependențe; un check verde este necesar, dar nu confirmă că inferența încape în GPU.
6. Pe profilul GitHub, **Packages → runpod-shorts-4090 → Package settings → Change visibility → Public**, dacă dorești ca RunPod să descarce imaginea fără credentiale GitHub. Publicarea imaginii trebuie făcută doar după ce confirmi că nu ai introdus secrete în fișiere; această arhivă nu include secrete.
7. Copiază adresa imaginii din rezumatul build-ului: `ghcr.io/CONTUL_TAU/runpod-shorts-4090:COMMIT_SHA`. Folosește tagul cu SHA, ca să păstrezi versiunea exactă.

## 2. Acces la modele și stocare

- Deschide https://huggingface.co/Lightricks/LTX-2 pe contul tău, citește și acceptă termenii dacă sunt solicitați. Acesta este LTX-2, nu modelul 2.5 vizitat anterior.
- Tokenul Hugging Face Read se introduce exclusiv în RunPod, ca secret/variabilă `HF_TOKEN`. Nu îl adăuga în GitHub, Dockerfile sau cererea de generare.
- În RunPod **Storage → Network Volume**, creează un volum de 100 GB într-un centru cu disponibilitate RTX 4090. Stocarea are cost separat și continuă să fie facturată între generări.
- Workerul descarcă aproximativ 34 GB de modele la prima cerere în cache-ul volumului. Descărcarea din primul job și încărcarea modelelor folosesc timp facturat. Următoarele porniri refolosesc fișierele; există totuși timp de încărcare în RAM/VRAM.

## 3. Creează endpointul

În **Serverless → New Endpoint → Custom container image / Import from Docker Registry** (denumirea poate varia):

| Setare | Valoare |
| --- | --- |
| Endpoint type | Queue-based |
| Name | shorts-4090 |
| Container image | Adresa GHCR cu SHA din build |
| GPU | RTX 4090 — 24 GB PRO |
| GPUs per worker | 1 |
| Active/minimum workers | 0 |
| Maximum workers | 1 |
| Idle timeout | 5 secunde |
| FlashBoot | Activat |
| Container disk | 30 GB, crește dacă imaginea nu încape |
| Network volume | Volumul de 100 GB, montat la /runpod-volume |
| CUDA | 12.8 sau mai nou |
| Execution timeout | 1800 secunde pentru primul test |
| Job TTL | 3600 secunde sau mai mult |
| Environment / secret | HF_TOKEN = tokenul tău Read |
| HTTP ports | Niciunul; ComfyUI ascultă doar pe localhost |
| Container start command | Lasă valoarea imaginii |

Limita de 1800 secunde permite descărcarea inițială, dar poate permite și un job scump. După primul test reușit, scade limita în funcție de durata măsurată. Max workers=1 limitează paralelismul, nu reprezintă buget maxim. Nu activa minimum=1 pentru a evita cold start-ul: asta facturează GPU-ul continuu.

## 4. Trimite promptul

Din panoul de testare al endpointului folosește conținutul din `test_input.json`, înlocuind doar promptul și, opțional, seed-ul:

```json
{
  "input": {
    "prompt": "SCRIE AICI PROMPTUL TAU",
    "seed": 42
  }
}
```

Pentru integrare externă: `POST https://api.runpod.ai/v2/ENDPOINT_ID/run`, apoi `GET https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID`, ambele cu header Authorization Bearer folosind cheia RunPod. Cheia RunPod este diferită de tokenul Hugging Face. Folosește `/run` asincron deoarece joburile video pot dura mult.

La `COMPLETED`, salvează JSON-ul răspunsului în `result.json`, apoi:

```sh
python3 decode_result.py result.json short.mp4
```

Fișierul MP4 este întors inline în `output.video_base64`. Descarcă rezultatul înainte de expirarea retenției RunPod (în mod obișnuit 30 minute pentru cereri asincrone); workerul șterge copiile temporare. Nu este configurat un bucket S3 separat.

## Ce generează

- Video + audio generate împreună cu LTX-2 distilled FP8 și encoder Gemma mixt FP4.
- O singură etapă, 8 pași, batch=1, low-VRAM și VAE tiled; fără un al doilea model de upscaling.
- 704×1280 și 241 cadre la 24 fps intern (dimensiuni compatibile cu modelul), apoi redimensionare cu raport păstrat și mic crop la **720×1280**, tăiere la **10 secunde**, H.264 + AAC.
- Adaugă instrucțiuni pentru ambianță și efecte, fără vorbire, șoapte, cântece sau vocalizări umane. Verifică sonor rezultatul; modelul poate încă produce voci nedorite.
- Nu rescrie promptul tău cu un LLM și nu publică videoclipuri.

## Primul test

Verifică: build reușit; modelul se descarcă; job COMPLETED; MP4 720×1280 de 10 secunde; sunet prezent și fără vorbire; costul efectiv și `executionTime`; workerul se oprește după golirea cozii. Dacă apare CUDA out-of-memory, oprește testele repetate și ajustează workflow-ul pe baza logului. Nu promite cost pe clip înainte de măsurare.

## Surse

- Workflow oficial: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_ltx2_t2v_distilled.json
- ComfyUI core v0.34.0: https://github.com/Comfy-Org/ComfyUI/tree/v0.34.0
- Model: https://huggingface.co/Lightricks/LTX-2
- Encoder: https://huggingface.co/Comfy-Org/ltx-2
- Setări RunPod: https://docs.runpod.io/serverless/endpoints/endpoint-configurations

Graful din `workflow.py` este o implementare proprie simplificată, bazată pe nodurile și programul sigma din workflow-ul oficial. Nu este exportul complet cu două etape al template-ului ComfyUI.
