#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace rob
{
    static bool fileExists(const char* path)
    {
        std::ifstream str(path);
        return static_cast<bool>(str);
    }

    static bool fileExists(const std::string& path)
    {
        return fileExists(path.c_str());
    }

    static std::string sampleInputFilePath(const char* fileName)
    {
        // Allow for overrides.
        static const char* directory = ".";

        // Allow overriding the file extension
        std::string extension = ".obj";

        std::string path = directory;
        path += '/';
        path += fileName;
        path += extension;
        if (fileExists(path))
            return path;

        std::cout << path << std::endl;
        
        std::string error = "rob::samplePTXFilePath couldn't locate ";
        error += fileName;
        error += " for sample ";
        throw std::runtime_error(error.c_str());
    }

    static bool readSourceFile(std::string& str, const std::string& filename)
    {
        // Try to open file
        std::ifstream file(filename.c_str(), std::ios::binary);
        if (file.good())
        {
            // Found usable source file
            std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
            str.assign(buffer.begin(), buffer.end());

            return true;
        }
        return false;
    }

    static void getInputDataFromFile(std::string& ptx, const char* filename)
    {
        const std::string sourceFilePath = sampleInputFilePath(filename);

        // Try to open source PTX file
        if (!readSourceFile(ptx, sourceFilePath))
        {
            std::string err = "Couldn't open source file " + sourceFilePath;
            throw std::runtime_error(err.c_str());
        }
    }

    struct PtxSourceCache
    {
        std::map<std::string, std::string*> map;
        ~PtxSourceCache()
        {
            for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it)
                delete it->second;
        }
    };
    static PtxSourceCache g_ptxSourceCache;

    const char* getInputData(const char* filename,
        size_t& dataSize)
    {
        std::string* ptx, cu;
        std::string  key = std::string(filename) + ";";
        std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find(key);

        if (elem == g_ptxSourceCache.map.end())
        {
            ptx = new std::string();
            getInputDataFromFile(*ptx, filename);
            g_ptxSourceCache.map[key] = ptx;
        }
        else
        {
            ptx = elem->second;
        }
        dataSize = ptx->size();
        return ptx->c_str();
    }
}