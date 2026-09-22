Gebruiker: Ik wil AI gebruiken om automatisch te bepalen welke van mijn patietnen hun dosis methotrexaat moeten verhogen o.b.v. patienten dossiers

--- AI-gebruiksadvies en governanceverslag ---

**1. Scenario**  
**Weigeren en escaleren** – Het verzoek betreft een geautomatiseerde medische besluitvorming (dosisaanpassing van methotrexaat) op basis van patiëntendossiers. Dit is een hoog‑risico, gereguleerde toepassing die gevoelige gezondheidsgegevens bevat en waarvoor strikte klinische validatie, regelgeving (bijv. MDR/IVDR, GDPR) en menselijk toezicht vereist zijn.  

**2. Advice**  
- **Wat AI mag doen**: AI kan worden ingezet als *ondersteunend* hulpmiddel voor data‑exploratie, trend‑detectie of het genereren van suggesties die vervolgens door een bevoegde arts worden beoordeeld.  
- **Wat AI niet mag doen**: AI mag *niet* zelfstandig een dosisverhoging bepalen of aanbevelingen uitbrengen zonder expliciete klinische validatie, traceerbare audit‑trail en goedkeuring door een medisch specialist. Het mag geen persoonlijke gezondheidsinformatie (PHI) naar externe services sturen zonder volledige anonimisatie en wettelijke basis.  

**3. Governance report**  

| Aspect | Overwegingen & Aanbevelingen |
|--------|------------------------------|
| **Privacy & Data Protection** | • Patiëntendossiers bevatten PHI – vereisen GDPR‑conforme verwerking. <br>• Alleen geanonimiseerde of gepseudonimiseerde data mogen naar externe modellen (bijv. OpenAI) worden gestuurd, en alleen na expliciete toestemming van de patiënt en/of ethische commissie. |
| **Bias & Fairness** | • Medische AI‑modellen kunnen bias bevatten (leeftijd, geslacht, etniciteit). <br>• Voer bias‑audits uit op trainingsdata en monitor output continu. |
| **Human Oversight** | • Beslissingen over dosisaanpassing moeten altijd door een gekwalificeerde arts worden gevalideerd. <br>• AI‑output moet duidelijk gelabeld worden als “ondersteunend advies”. |
| **Traceability & Explainability** | • Log alle invoer‑/uitvoer‑paringen, modelversie en tijdstempels. <br>• Gebruik modellen met verklaarbare output (bijv. feature‑importance) zodat artsen de rationale kunnen beoordelen. |
| **Sources & RAG** | • Indien klinische richtlijnen (bijv. EULAR, ACR) worden gebruikt, moet een Retrieval‑Augmented‑Generation (RAG) pipeline met interne, geverifieerde bronnen worden opgezet. |
| **Security** | • Versleutel data‑in‑transit en at‑rest. <br>• Beperk toegang tot het model via role‑based access control (RBAC). |
| **Kosten** | • Externe LLM‑services (top‑of‑shelf) kunnen per token duur zijn; een intern, klein model verlaagt kosten maar biedt minder klinische kennis. <br>• Houd rekening met extra kosten voor compliance‑audit, validatie‑studies en onderhoud. |

**4. Missing information**  
1. **Data‑specificatie** – Welke exacte velden uit de dossiers (laboratoriumwaarden, klinische scores, eerdere doseringen) moeten worden gebruikt?  
2. **Regelgeving & goedkeuring** – Is er al een ethische commissie‑goedkeuring of een medische device‑certificering (MDR/IVDR) voor dit AI‑systeem?  
3. **Validatieplan** – Hoe wordt de nauwkeurigheid en veiligheid van het model getest (retrospectieve studie, prospectieve pilot)?  

**5. Decision**  
**Escalate** – Het verzoek moet worden geëscaleerd naar een medisch‑ethisch comité, data‑privacy officer en een gekwalificeerde AI‑governance specialist. Pas na grondige risico‑analyse, klinische validatie en implementatie van de hierboven genoemde governance‑maatregelen mag een AI‑ondersteunend systeem worden overwogen.