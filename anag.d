/*
Example from wikipedia

http://en.wikipedia.org/wiki/D_%28programming_language%29#Example_2
*/

import std.stdio, std.algorithm, std.range, std.string;
 
void main()
{
    dstring[][dstring] signs2words;
 
    foreach(dchar[] w; lines(File("words.txt")))
    {
        w = w.chomp().toLower();
        immutable key = w.dup.sort().release().idup;
        signs2words[key] ~= w.idup;
    }
 
    foreach(words; signs2words)
        if(words.length > 1)
            writefln(words.join(" "));
}
