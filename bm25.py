from typing import List
import math

TEXT = """
The Art and Science of Cooking: A Deep Dive into Culinary Creativity

Cooking is one of humanity’s oldest and most universal activities. From the discovery of fire to the rise of molecular gastronomy, it has evolved from a basic survival skill into an intricate form of art and science. While every culture has its unique traditions, the essence of cooking—transforming raw ingredients into something nourishing and enjoyable—remains constant. Behind every dish lies a blend of chemistry, physics, creativity, and emotion. Understanding cooking means understanding not just recipes but the deep processes that make flavors come alive.

At its core, cooking is chemistry. When you apply heat to food, you are essentially initiating chemical reactions. Proteins denature and coagulate, starches gelatinize, and sugars caramelize. These processes create the textures, colors, and aromas we associate with cooked food. For instance, the Maillard reaction, which occurs when amino acids and reducing sugars interact under heat, gives grilled meat its rich brown crust and roasted coffee its deep aroma. It is one of the most celebrated phenomena in culinary science because it enhances flavor complexity and visual appeal.

Equally important is the concept of balance. Great chefs know that flavor is a multidimensional experience involving taste, aroma, texture, and even sound. The five basic tastes—sweet, sour, salty, bitter, and umami—must be carefully balanced to create harmony in a dish. Salt, for example, not only enhances flavor but also suppresses bitterness and highlights sweetness. Acid, often from lemon juice or vinegar, cuts through richness and refreshes the palate. Umami, the savory taste found in ingredients like tomatoes, mushrooms, and soy sauce, adds depth and satisfaction.

Cooking also involves a deep understanding of ingredients. Freshness, quality, and seasonality are critical factors. A ripe tomato in summer tastes entirely different from one grown in a greenhouse in winter. Understanding where ingredients come from and how they behave under different conditions allows cooks to make better choices. For example, knowing that starches in potatoes vary by variety helps determine whether a potato is best for mashing, frying, or roasting. Similarly, the fat content of milk or butter affects texture in baking and sauces.

Another key aspect of cooking is technique. Even the simplest dish can be elevated by the right technique. Knife skills, for instance, affect both presentation and cooking time. Uniformly cut vegetables cook evenly and look more appealing. Techniques like sautéing, roasting, braising, and steaming each have distinct purposes and effects. Sautéing quickly cooks ingredients over high heat, preserving freshness and color, while braising uses low, slow heat to tenderize tough cuts of meat and develop rich flavors.

Cultural diversity in cooking is immense. Each cuisine represents centuries of adaptation to geography, climate, and available resources. Italian cuisine emphasizes simplicity and fresh ingredients—olive oil, tomatoes, garlic, and herbs—while French cuisine is known for its sauces and precision. Indian cuisine, with its complex use of spices, creates layers of flavor that engage all senses. Japanese cooking celebrates seasonality and minimalism, often highlighting the natural flavor of ingredients with minimal interference. Exploring these cuisines not only broadens culinary knowledge but also provides insights into the history and identity of different peoples.

Cooking is also a deeply emotional and social act. Preparing food for others expresses care and love. Family recipes are passed down through generations, carrying memories and traditions. The smell of a familiar dish can transport someone back to childhood or remind them of a particular person or place. In many cultures, food plays a central role in rituals and celebrations—Thanksgiving in the United States, Lunar New Year feasts in China, or Diwali sweets in India. These shared meals strengthen social bonds and provide comfort.

In recent decades, the relationship between cooking and health has become increasingly important. With the rise of processed foods and sedentary lifestyles, people are rediscovering the benefits of home cooking. When you cook your own meals, you control the ingredients, portion sizes, and cooking methods. Studies have shown that people who cook at home tend to eat more vegetables, consume fewer calories, and have better overall nutrition. Moreover, cooking can be therapeutic. The act of chopping vegetables, kneading dough, or stirring a simmering pot provides mindfulness and a sense of accomplishment.

Technology has transformed cooking as well. From induction cooktops to smart ovens, modern tools make precision and consistency easier to achieve. Sous-vide cooking, for example, allows chefs to cook food at exact temperatures in water baths, ensuring perfect doneness every time. High-speed blenders and food processors save time, while 3D food printing is pushing the boundaries of creativity. Online platforms, cooking blogs, and video tutorials have made culinary education more accessible than ever. A home cook can now learn techniques that were once reserved for professional chefs.

However, despite all the advancements, cooking remains a profoundly human activity. Machines can replicate precision, but they cannot reproduce the intuition and emotion that a human brings to the kitchen. The way a cook adjusts seasoning by taste or decides when something “feels right” cannot be programmed. This blend of instinct and technique is what makes cooking both an art and a science.

Sustainability is another growing theme in modern cooking. As awareness of environmental issues increases, many chefs and home cooks are reconsidering how they source ingredients. Farm-to-table movements emphasize local and seasonal produce, reducing carbon footprints and supporting local farmers. Reducing food waste has become a key priority—using vegetable scraps for stocks, repurposing leftovers, and finding creative ways to use every part of an ingredient. Plant-based cooking is also gaining popularity, not only for health reasons but also as a response to the environmental impact of animal agriculture.

Baking, a distinct branch of cooking, is often described as a precise science. Unlike savory cooking, where improvisation is encouraged, baking requires exact measurements and timing. The proportions of flour, fat, sugar, and liquid determine the texture and structure of baked goods. Leavening agents like yeast, baking powder, or baking soda introduce gases that make dough rise. Temperature control is critical—too hot, and a cake burns on the outside while staying raw inside; too cold, and it may never rise properly. Yet even in baking, creativity has a place. Bakers experiment with flavors, decorations, and innovative ingredients like gluten-free flours or plant-based substitutes.

Cooking is also about learning and experimentation. Mistakes are inevitable, but they are also opportunities to grow. Burning a sauce teaches you about heat control; overseasoning a dish teaches restraint. Some of the world’s greatest culinary inventions came from accidents—chocolate chip cookies, potato chips, even champagne. The kitchen is a laboratory where curiosity leads to discovery.

Culinary traditions are constantly evolving. Globalization has blended cuisines in unprecedented ways. Fusion food—mixing techniques and ingredients from different cultures—has become a creative playground. Sushi burritos, Korean tacos, and matcha croissants are examples of how boundaries can be redefined. While purists may criticize fusion as inauthentic, it reflects the natural evolution of culture and taste.

At the same time, there’s a growing appreciation for traditional cooking methods that risk being forgotten. Slow fermentation, wood-fired baking, and hand-grinding spices are being revived by chefs who value authenticity and craftsmanship. These methods connect people to the roots of culinary heritage and remind us that patience and care are timeless ingredients.

Ultimately, cooking is about connection—between people, between cultures, and between humans and nature. It invites creativity, nurtures community, and celebrates life’s simple pleasures. Whether it’s a humble bowl of soup or a gourmet tasting menu, every meal tells a story. Cooking transforms the ordinary into the extraordinary, turning raw materials into moments of joy, comfort, and discovery.

In the words of the legendary chef Julia Child, “No one is born a great cook; one learns by doing.” Cooking is a lifelong journey of learning, tasting, failing, and improving. It’s a dance between control and spontaneity, precision and intuition. And in that delicate balance lies the true beauty of the culinary arts.
"""



def tf(keyword: str, doc: str) -> float:
    keyword = keyword.strip().lower()
    words = [word.lower() for word in doc.split()]
    num_words = len(words)
    num_of_keyword = sum(1 for word in words if keyword == word)
    #print(f"keyword({keyword}) occured {num_of_keyword} in text that has {num_words} words")
    return num_of_keyword / num_words if num_words > 0 else 0.0

def idf(keyword: str, docs: List[str]) -> float:
    keyword = keyword.strip().lower()
    num_of_doc_with_keyword = sum(1 for doc in docs if keyword in [word.lower() for word in doc.split()])
    return math.log((len(docs) - num_of_doc_with_keyword + 0.5) / (num_of_doc_with_keyword + 0.5))

def bm25(keyword: str, doc: str, docs: List[str], k1: float =1.5, b: float =0.75) -> float:
    avg_len = sum(len(doc.split()) for doc in docs) / len(docs)
    keyword = keyword.strip()
    num_words_in_doc = len(doc.split())
    tf_val = tf(keyword, doc)

    numerator = idf(keyword, docs) * tf_val * (k1 + 1)
    denominator = tf_val + k1 * (1 - b + b * (float(num_words_in_doc / avg_len)))

    return numerator / denominator if denominator != 0 else 0.0

docs = [TEXT]
keyword = "globalization"
print(f"Results for keyword: {keyword}")
print("TF: ", tf(keyword, TEXT))
print("IDF: ", idf(keyword, docs))
print("BM25: ", bm25(keyword, TEXT, docs))

# Implemented by lukl123
