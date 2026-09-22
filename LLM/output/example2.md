AI-verzoek: 
Voorbeeldverzoek gebruikt: Beoordeel of AI een fondsaanvraag mag samenvatten, controleren op volledigheid en aandachtspunten kan markeren voor een medewerker.

--- AI-gebruiksadvies en governanceverslag ---

**1. Scenario**  
**RAG (Retrieval‑Augmented Generation)** – gebruik interne beleids‑ en checklist‑documenten als bron en een intern (klein) generiek LLM om de fondsaanvraag te samenvatten, op volledigheid te toetsen en aandachtspunten te markeren.  

**2. Advice**  
- **Wat AI mag doen**  
  - De volledige tekst van de fondsaanvraag (in‑house) inlezen.  
  - Een beknopte samenvatting (max. 200 woorden) genereren.  
  - De inhoud vergelijken met de interne checklist / richtlijnen (bijv. projectbeschrijving, begroting, risico‑analyse, juridische compliance).  
  - Ontbrekende of onduidelijke onderdelen markeren en een korte toelichting geven.  
- **Wat AI niet mag doen**  
  - De inhoud delen met externe systemen of cloud‑providers.  
  - Definitieve beslissingen nemen over toekenning of afwijzing.  
  - Persoonlijke of vertrouwelijke gegevens (bijv. namen, bankrekeningen) buiten de beveiligde omgeving verwerken of opslaan.  

**3. Governance report**  

| Aspect | Aanbeveling |
|--------|-------------|
| **Privacy & Data‑protectie** | Verwerk de aanvraag uitsluitend binnen de beveiligde interne omgeving (OCI‑cluster). Anonimiseer of redacteer persoonsgegevens voordat ze aan het model worden gevoed. |
| **Bias & Fairness** | Gebruik een neutraal, klein generiek LLM; controleer output op mogelijke bias (bijv. onbedoelde nadruk op bepaalde sectoren). Laat een mens de gemarkeerde punten valideren. |
| **Human Oversight** | Een medewerker moet de samenvatting en de checklist‑resultaten goedkeuren voordat ze worden verspreid. |
| **Traceability** | Log alle prompts, model‑versies en gebruikte interne documenten (metadata). Bewaar logs 90 dagen voor audit. |
| **Bronnen** | Interne beleids‑ en checklist‑documenten (bijv. “Fondsaanvraag‑richtlijn v2.3”). Vermeld in de output welke bron voor elk gemarkeerd punt is gebruikt. |
| **Security** | Zorg voor netwerk‑isolatie (geen internettoegang), versleuteling in rust en tijdens transport, en rol‑gebaseerde toegangscontrole. |
| **Kosten** | Klein intern LLM (≈ 0,02 USD per 1 000 tokens) + opslag‑/retrieval‑kosten (≈ 0,005 USD per 1 000 tokens). Voor een gemiddelde aanvraag van 5 000 tokens verwacht € 0,15 – 0,30 per aanvraag. |

**4. Missing information**  
1. **Welke interne checklist‑/richtlijndocumenten moeten worden gebruikt?** (bijv. “Financiële‑verantwoording v1.0”).  
2. **Hoe gevoelig is de informatie in de fondsaanvraag?** (bijv. bevat het persoonsgegevens, bedrijfsgeheimen?).  
3. **Wat is het gewenste output‑formaat?** (plain‑text, PDF‑annotatie, gestructureerde JSON?).  

**5. Decision**  
**Approve with conditions** – AI‑ondersteuning mag worden ingezet onder de hierboven gespecificeerde RAG‑opzet, met strikte interne verwerking, menselijke eindcontrole en volledige logging. Zodra de ontbrekende informatie is aangeleverd, kan de implementatie worden gestart.