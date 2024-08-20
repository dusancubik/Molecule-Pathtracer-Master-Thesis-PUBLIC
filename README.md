# Molecule-Pathtracer-Master-Thesis-PUBLIC
## Cíl: Vyrenderovat větší množství(50k) primitiv pomocí progresivního pathtracingu.
1. prozkoumal jsem různé akcelerační struktury a jejich varianty
2. řešil layout pro BVH uzly v GPU bufferu pro efektivnější cache hit
3. Zachování realtime - 1 shader = 1 ray bounce -> Shader nepočítá celou cestu, ale vždy jen jeden odraz a další odraz počítá jiná instance Shader programu (potřebné informace se uloží do textur). Pokud by Shader instance počítala celou cestu (např. 10 odrazů), tak pokud by chtěl uživatel přemístit kameru, tak by musel počkat.


## TODO:
- vyřešit fireflies
- vyřešit mem. leaky
- macro pro cesty k souborům
- UI - výběr proteinu, barvy, skybox

High Roughness: https://youtu.be/Qd9NzgdKY5w
Zero Roughness: 