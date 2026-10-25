"""Offline place gazetteer for instant autocomplete (countries → capital coordinates, plus major cities).
Live suggestions from the free Open-Meteo geocoding API are merged in by the server when online."""
from __future__ import annotations

import unicodedata

# Country | capital | lat | lon | ISO-2
_COUNTRIES = """Afghanistan|Kabul|34.53|69.17|AF
Albania|Tirana|41.33|19.82|AL
Algeria|Algiers|36.75|3.06|DZ
Andorra|Andorra la Vella|42.51|1.52|AD
Angola|Luanda|-8.84|13.23|AO
Argentina|Buenos Aires|-34.60|-58.38|AR
Armenia|Yerevan|40.18|44.51|AM
Australia|Canberra|-35.28|149.13|AU
Austria|Vienna|48.21|16.37|AT
Azerbaijan|Baku|40.41|49.87|AZ
Bahamas|Nassau|25.05|-77.35|BS
Bahrain|Manama|26.23|50.59|BH
Bangladesh|Dhaka|23.81|90.41|BD
Barbados|Bridgetown|13.10|-59.62|BB
Belarus|Minsk|53.90|27.56|BY
Belgium|Brussels|50.85|4.35|BE
Belize|Belmopan|17.25|-88.77|BZ
Benin|Porto-Novo|6.50|2.60|BJ
Bhutan|Thimphu|27.47|89.64|BT
Bolivia|La Paz|-16.49|-68.12|BO
Bosnia and Herzegovina|Sarajevo|43.86|18.41|BA
Botswana|Gaborone|-24.65|25.91|BW
Brazil|Brasília|-15.79|-47.88|BR
Brunei|Bandar Seri Begawan|4.90|114.94|BN
Bulgaria|Sofia|42.70|23.32|BG
Burkina Faso|Ouagadougou|12.37|-1.52|BF
Burundi|Gitega|-3.43|29.93|BI
Cambodia|Phnom Penh|11.56|104.92|KH
Cameroon|Yaoundé|3.85|11.50|CM
Canada|Ottawa|45.42|-75.70|CA
Cape Verde|Praia|14.93|-23.51|CV
Central African Republic|Bangui|4.39|18.56|CF
Chad|N'Djamena|12.13|15.06|TD
Chile|Santiago|-33.45|-70.67|CL
China|Beijing|39.90|116.41|CN
Colombia|Bogotá|4.71|-74.07|CO
Comoros|Moroni|-11.70|43.26|KM
Congo|Brazzaville|-4.26|15.24|CG
Costa Rica|San José|9.93|-84.08|CR
Croatia|Zagreb|45.81|15.98|HR
Cuba|Havana|23.11|-82.37|CU
Cyprus|Nicosia|35.19|33.38|CY
Czechia|Prague|50.08|14.44|CZ
Democratic Republic of the Congo|Kinshasa|-4.44|15.27|CD
Denmark|Copenhagen|55.68|12.57|DK
Djibouti|Djibouti|11.59|43.15|DJ
Dominican Republic|Santo Domingo|18.49|-69.93|DO
Ecuador|Quito|-0.18|-78.47|EC
Egypt|Cairo|30.04|31.24|EG
El Salvador|San Salvador|13.69|-89.22|SV
Equatorial Guinea|Malabo|3.75|8.78|GQ
Eritrea|Asmara|15.32|38.93|ER
Estonia|Tallinn|59.44|24.75|EE
Eswatini|Mbabane|-26.31|31.14|SZ
Ethiopia|Addis Ababa|9.03|38.74|ET
Fiji|Suva|-18.14|178.44|FJ
Finland|Helsinki|60.17|24.94|FI
France|Paris|48.86|2.35|FR
Gabon|Libreville|0.42|9.47|GA
Gambia|Banjul|13.45|-16.58|GM
Georgia|Tbilisi|41.72|44.79|GE
Germany|Berlin|52.52|13.40|DE
Ghana|Accra|5.60|-0.19|GH
Greece|Athens|37.98|23.73|GR
Guatemala|Guatemala City|14.63|-90.51|GT
Guinea|Conakry|9.64|-13.58|GN
Guyana|Georgetown|6.80|-58.16|GY
Haiti|Port-au-Prince|18.59|-72.31|HT
Honduras|Tegucigalpa|14.07|-87.19|HN
Hungary|Budapest|47.50|19.04|HU
Iceland|Reykjavík|64.15|-21.94|IS
India|New Delhi|28.61|77.21|IN
Indonesia|Jakarta|-6.21|106.85|ID
Iran|Tehran|35.69|51.39|IR
Iraq|Baghdad|33.31|44.37|IQ
Ireland|Dublin|53.35|-6.26|IE
Israel|Jerusalem|31.77|35.21|IL
Italy|Rome|41.90|12.50|IT
Ivory Coast|Yamoussoukro|6.83|-5.29|CI
Jamaica|Kingston|17.97|-76.79|JM
Japan|Tokyo|35.68|139.65|JP
Jordan|Amman|31.95|35.93|JO
Kazakhstan|Astana|51.17|71.45|KZ
Kenya|Nairobi|-1.29|36.82|KE
Kuwait|Kuwait City|29.38|47.99|KW
Kyrgyzstan|Bishkek|42.87|74.59|KG
Laos|Vientiane|17.98|102.63|LA
Latvia|Riga|56.95|24.11|LV
Lebanon|Beirut|33.89|35.50|LB
Lesotho|Maseru|-29.31|27.48|LS
Liberia|Monrovia|6.30|-10.80|LR
Libya|Tripoli|32.89|13.19|LY
Lithuania|Vilnius|54.69|25.28|LT
Luxembourg|Luxembourg|49.61|6.13|LU
Madagascar|Antananarivo|-18.88|47.51|MG
Malawi|Lilongwe|-13.96|33.79|MW
Malaysia|Kuala Lumpur|3.14|101.69|MY
Maldives|Malé|4.18|73.51|MV
Mali|Bamako|12.64|-8.00|ML
Malta|Valletta|35.90|14.51|MT
Mauritania|Nouakchott|18.07|-15.96|MR
Mauritius|Port Louis|-20.16|57.50|MU
Mexico|Mexico City|19.43|-99.13|MX
Moldova|Chișinău|47.01|28.86|MD
Mongolia|Ulaanbaatar|47.89|106.91|MN
Montenegro|Podgorica|42.43|19.26|ME
Morocco|Rabat|34.02|-6.83|MA
Mozambique|Maputo|-25.97|32.57|MZ
Myanmar|Naypyidaw|19.76|96.08|MM
Namibia|Windhoek|-22.56|17.08|NA
Nepal|Kathmandu|27.72|85.32|NP
Netherlands|Amsterdam|52.37|4.90|NL
New Zealand|Wellington|-41.29|174.78|NZ
Nicaragua|Managua|12.11|-86.24|NI
Niger|Niamey|13.51|2.13|NE
Nigeria|Abuja|9.08|7.40|NG
North Korea|Pyongyang|39.04|125.76|KP
North Macedonia|Skopje|41.99|21.43|MK
Norway|Oslo|59.91|10.75|NO
Oman|Muscat|23.59|58.41|OM
Pakistan|Islamabad|33.68|73.05|PK
Palestine|Ramallah|31.90|35.20|PS
Panama|Panama City|8.98|-79.52|PA
Papua New Guinea|Port Moresby|-9.44|147.18|PG
Paraguay|Asunción|-25.26|-57.58|PY
Peru|Lima|-12.05|-77.04|PE
Philippines|Manila|14.60|120.98|PH
Poland|Warsaw|52.23|21.01|PL
Portugal|Lisbon|38.72|-9.14|PT
Qatar|Doha|25.29|51.53|QA
Romania|Bucharest|44.43|26.10|RO
Russia|Moscow|55.76|37.62|RU
Rwanda|Kigali|-1.94|30.06|RW
Saudi Arabia|Riyadh|24.71|46.68|SA
Senegal|Dakar|14.72|-17.47|SN
Serbia|Belgrade|44.79|20.45|RS
Sierra Leone|Freetown|8.48|-13.23|SL
Singapore|Singapore|1.35|103.82|SG
Slovakia|Bratislava|48.15|17.11|SK
Slovenia|Ljubljana|46.06|14.51|SI
Somalia|Mogadishu|2.05|45.32|SO
South Africa|Pretoria|-25.75|28.19|ZA
South Korea|Seoul|37.57|126.98|KR
South Sudan|Juba|4.85|31.58|SS
Spain|Madrid|40.42|-3.70|ES
Sri Lanka|Colombo|6.93|79.86|LK
Sudan|Khartoum|15.50|32.56|SD
Suriname|Paramaribo|5.85|-55.20|SR
Sweden|Stockholm|59.33|18.07|SE
Switzerland|Bern|46.95|7.45|CH
Syria|Damascus|33.51|36.28|SY
Taiwan|Taipei|25.03|121.57|TW
Tajikistan|Dushanbe|38.56|68.79|TJ
Tanzania|Dodoma|-6.16|35.75|TZ
Thailand|Bangkok|13.76|100.50|TH
Togo|Lomé|6.13|1.22|TG
Trinidad and Tobago|Port of Spain|10.65|-61.51|TT
Tunisia|Tunis|36.81|10.18|TN
Turkey|Ankara|39.93|32.86|TR
Turkmenistan|Ashgabat|37.96|58.33|TM
Uganda|Kampala|0.35|32.58|UG
Ukraine|Kyiv|50.45|30.52|UA
United Arab Emirates|Abu Dhabi|24.45|54.38|AE
United Kingdom|London|51.51|-0.13|GB
United States|Washington, D.C.|38.91|-77.04|US
Uruguay|Montevideo|-34.90|-56.16|UY
Uzbekistan|Tashkent|41.30|69.24|UZ
Venezuela|Caracas|10.48|-66.90|VE
Vietnam|Hanoi|21.03|105.85|VN
Yemen|Sanaa|15.37|44.19|YE
Zambia|Lusaka|-15.39|28.32|ZM
Zimbabwe|Harare|-17.83|31.05|ZW"""

# City | region/country | lat | lon | ISO-2
_CITIES = """Istanbul|Türkiye|41.01|28.98|TR
Izmir|Türkiye|38.42|27.14|TR
Antalya|Türkiye|36.90|30.71|TR
Bursa|Türkiye|40.19|29.06|TR
Adana|Türkiye|37.00|35.32|TR
Konya|Türkiye|37.87|32.48|TR
Trabzon|Türkiye|41.00|39.72|TR
Gaziantep|Türkiye|37.07|37.38|TR
Mashhad|Iran|36.30|59.61|IR
Isfahan|Iran|32.65|51.67|IR
Shiraz|Iran|29.59|52.58|IR
Tabriz|Iran|38.08|46.29|IR
Ahvaz|Iran|31.32|48.67|IR
Rasht|Iran|37.28|49.58|IR
Bandar Abbas|Iran|27.18|56.27|IR
Kerman|Iran|30.28|57.08|IR
Yazd|Iran|31.90|54.37|IR
Basra|Iraq|30.51|47.78|IQ
Erbil|Iraq|36.19|44.01|IQ
San Francisco|California, US|37.77|-122.42|US
San Jose|California, US|37.34|-121.89|US
San Diego|California, US|32.72|-117.16|US
San Antonio|Texas, US|29.42|-98.49|US
Sacramento|California, US|38.58|-121.49|US
Los Angeles|California, US|34.05|-118.24|US
Fresno|California, US|36.74|-119.79|US
Santa Barbara|California, US|34.42|-119.70|US
Santa Cruz|California, US|36.97|-122.03|US
Oakland|California, US|37.80|-122.27|US
Palo Alto|California, US|37.44|-122.14|US
Seattle|Washington, US|47.61|-122.33|US
Portland|Oregon, US|45.52|-122.68|US
Las Vegas|Nevada, US|36.17|-115.14|US
Phoenix|Arizona, US|33.45|-112.07|US
Denver|Colorado, US|39.74|-104.99|US
Dallas|Texas, US|32.78|-96.80|US
Houston|Texas, US|29.76|-95.37|US
Austin|Texas, US|30.27|-97.74|US
Chicago|Illinois, US|41.88|-87.63|US
New York|New York, US|40.71|-74.01|US
Boston|Massachusetts, US|42.36|-71.06|US
Miami|Florida, US|25.76|-80.19|US
New Orleans|Louisiana, US|29.95|-90.07|US
Atlanta|Georgia, US|33.75|-84.39|US
Honolulu|Hawaii, US|21.31|-157.86|US
Anchorage|Alaska, US|61.22|-149.90|US
Toronto|Canada|43.65|-79.38|CA
Vancouver|Canada|49.28|-123.12|CA
Montreal|Canada|45.50|-73.57|CA
São Paulo|Brazil|-23.55|-46.63|BR
Rio de Janeiro|Brazil|-22.91|-43.17|BR
Manaus|Brazil|-3.12|-60.02|BR
Santiago|Chile|-33.45|-70.67|CL
Bogotá|Colombia|4.71|-74.07|CO
Guadalajara|Mexico|20.67|-103.35|MX
Monterrey|Mexico|25.69|-100.32|MX
Barcelona|Spain|41.39|2.17|ES
Valencia|Spain|39.47|-0.38|ES
Seville|Spain|37.39|-5.98|ES
Milan|Italy|45.46|9.19|IT
Naples|Italy|40.85|14.27|IT
Venice|Italy|45.44|12.32|IT
Munich|Germany|48.14|11.58|DE
Hamburg|Germany|53.55|9.99|DE
Frankfurt|Germany|50.11|8.68|DE
Marseille|France|43.30|5.37|FR
Lyon|France|45.76|4.84|FR
Manchester|United Kingdom|53.48|-2.24|GB
Edinburgh|United Kingdom|55.95|-3.19|GB
Zurich|Switzerland|47.38|8.54|CH
Geneva|Switzerland|46.20|6.14|CH
Rotterdam|Netherlands|51.92|4.48|NL
Thessaloniki|Greece|40.64|22.94|GR
Saint Petersburg|Russia|59.93|30.34|RU
Dubai|United Arab Emirates|25.20|55.27|AE
Jeddah|Saudi Arabia|21.49|39.19|SA
Mumbai|India|19.08|72.88|IN
Kolkata|India|22.57|88.36|IN
Chennai|India|13.08|80.27|IN
Bengaluru|India|12.97|77.59|IN
Karachi|Pakistan|24.86|67.01|PK
Lahore|Pakistan|31.55|74.34|PK
Shanghai|China|31.23|121.47|CN
Guangzhou|China|23.13|113.26|CN
Shenzhen|China|22.54|114.06|CN
Hong Kong|China|22.32|114.17|HK
Wuhan|China|30.59|114.31|CN
Osaka|Japan|34.69|135.50|JP
Sapporo|Japan|43.06|141.35|JP
Busan|South Korea|35.18|129.08|KR
Ho Chi Minh City|Vietnam|10.82|106.63|VN
Sydney|Australia|-33.87|151.21|AU
Melbourne|Australia|-37.81|144.96|AU
Brisbane|Australia|-27.47|153.03|AU
Perth|Australia|-31.95|115.86|AU
Auckland|New Zealand|-36.85|174.76|NZ
Lagos|Nigeria|6.52|3.38|NG
Cape Town|South Africa|-33.92|18.42|ZA
Johannesburg|South Africa|-26.20|28.05|ZA
Casablanca|Morocco|33.57|-7.59|MA
Alexandria|Egypt|31.20|29.92|EG
Antalya Coast (Alanya)|Türkiye|36.54|32.00|TR"""


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def _flag(iso: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso.upper()) if len(iso) == 2 and iso.isalpha() else "📍"


PLACES: list[dict] = []
for line in _COUNTRIES.splitlines():
    n, cap, la, lo, iso = line.split("|")
    PLACES.append({"name": n, "detail": f"country · capital {cap}", "lat": float(la), "lon": float(lo), "kind": "country", "flag": _flag(iso)})
    PLACES.append({"name": cap, "detail": f"capital of {n}", "lat": float(la), "lon": float(lo), "kind": "city", "flag": _flag(iso)})
for line in _CITIES.splitlines():
    n, reg, la, lo, iso = line.split("|")
    PLACES.append({"name": n, "detail": reg, "lat": float(la), "lon": float(lo), "kind": "city", "flag": _flag(iso)})
_seen = set()
PLACES = [p for p in PLACES if not ((p["name"], round(p["lat"], 1)) in _seen or _seen.add((p["name"], round(p["lat"], 1))))]


def search(q: str, limit: int = 8) -> list[dict]:
    qn = _norm(q.strip())
    if not qn:
        return []
    scored = []
    for p in PLACES:
        n = _norm(p["name"])
        words = n.replace(",", " ").split()
        if n.startswith(qn):
            s = 0
        elif any(w.startswith(qn) for w in words):
            s = 1
        elif qn in n:
            s = 2
        elif qn in _norm(p["detail"]):
            s = 3
        else:
            continue
        exact = 0 if p["name"].lower() == q.strip().lower() else 1      # exact spelling first: "San Jose" → California, "San José" → Costa Rica
        scored.append((s, exact, 0 if p["kind"] == "country" and len(qn) <= 3 else 1, len(n), p))
    scored.sort(key=lambda x: x[:4])
    return [x[4] for x in scored[:limit]]
