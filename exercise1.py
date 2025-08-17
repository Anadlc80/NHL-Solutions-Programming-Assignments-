# --------------------------------------------------------------------------------------------------------------------------
# Ana de la Cerda Galvan ------- Exercise 1 - NHL REQUERIMENTS  ------------------------------------------------------------
# There is some Spanish code inside, I introduced them while I studied the code to be clear of what id does en every step
#---------------------------------------------------------------------------------------------------------------------------
#  BUG
# The problem in this exercise was the type of structure used. The programm try to use an index in a Set estructure
# The Set estructure doesn't have any specific orden and we can't access to its elems with an index. As result every time that we call the function we will have an aleatory fruit.
# My solution trying to keep the same structure of the original program. I only convert it to an ordened list. After that with a loop I will look for the same index.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPROVE
# But for me, I will use a more efficient way, I would just convert the set in an orden list and access to it directly with ther index. So we wouldn't need the loop at all.
#      lista_ordenada=sorted(fruits)
#      return lista_ordenada[fruit_id]
# I wasn't sure how much I was allowed to change the code, but I let you know the both options.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def id_to_fruit(fruit_id: int, fruits:[str]) -> str:
    idx = 0
    for fruit in sorted(fruits):
        if fruit_id == idx:
            return fruit
        idx += 1
    raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

if __name__=="__main__":
    name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
    name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
    name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    name5 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    print(name1,name3,name4,name5)
