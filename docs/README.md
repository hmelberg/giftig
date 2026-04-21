# Gift-API test-UI

Statisk webside som kaller gift-API-et på `https://giftig.anvil.app/_/api/ask`
direkte fra nettleseren. Ligger her så den kan serveres via GitHub Pages uten
å påvirke Anvil-sync på master.

## Åpne siden

**Lokalt:** dobbeltklikk `index.html`, eller start en enkel server:

```
python -m http.server 8765
```

og åpne `http://localhost:8765/`.

**Via GitHub Pages (når aktivert):**
Repo → **Settings → Pages → Source: Deploy from a branch → Branch: `master`,
Folder: `/docs`**. Siden dukker opp på
`https://hmelberg.github.io/giftig/` innen ~1 minutt.

## Første gang

1. Klikk "Innstillinger" øverst til høyre
2. Lim inn API-nøkkelen din i "API-nøkkel"-feltet (lagres i `localStorage`)
3. Still et spørsmål, eller klikk ett av eksemplene

## Viktig

**Testversjon.** Se det røde banneret øverst på siden. Ikke offisiell
informasjon fra FHI/Helsedirektoratet/Giftinformasjonen. Ikke bruk som
medisinsk rådgivning.
