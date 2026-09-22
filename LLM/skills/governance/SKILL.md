# Onderwerp
Evaluate request for AI usage

# Wat
Beoordeel elk AI-verzoek op vraagtype, benodigde informatie, risico's, privacy, ethiek en kosten. Kies daarna één scenario en maak kort een adviesrapport met aandacht voor data- en AI-governance:

1. **RAG**: gebruik interne bronnen wanneer het antwoord daarop moet zijn gebaseerd. Vermeld de gebruikte bronnen.
2. **Top-of-shelf LLM**: gebruik een krachtig extern model, bijvoorbeeld OpenAI of Anthropic, voor complexe redenering of actuele webinformatie.
3. **Klein generiek LLM**: gebruik een licht model binnen OCI voor eenvoudige, algemene vragen zonder externe informatie.
4. **Weigeren en escaleren**: wijs schadelijke, niet-toegestane of uitzonderlijk dure verzoeken af en leg ze voor aan een mens.

Als de informatie ontbreekt, stel een keuzevraag met 2 of 3 concrete opties plus 'iets anders, namelijk...'.

# Wanneer gebruiken
Zodra een gebruiker een verzoek indient waarbij AI of een LLM kan worden gebruikt.

# Voorbeeld
Een gebruiker wil een e-mail netter formuleren. Gebruik het kleine generieke OCI-model: er is geen webinformatie nodig en de tekst bevat geen gevoelige gegevens. Er is daarom geen RAG of extern model nodig.
