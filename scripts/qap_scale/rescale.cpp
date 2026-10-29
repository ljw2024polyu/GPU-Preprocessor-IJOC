// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

// Column rescaling for Hinder's symmetry-reduced Adams-Johnson MPS output.
// This uses the paper generator's scale: 1/abs(c_j) for nonzero cost, 1 otherwise.
// Rows have RHS in {0, 1}; variables are nonnegative and have no upper bounds.
// Source formulation: ohinder/large-scale-LP-test-problems, commit
// b44862990e19ddaeab5568a826bd4ced13f23819 (newlp-300.f and utils.jl).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Reader {
    FILE* f;
    char line[2048];
    char* t[8];
    int nt = 0;
    long long lines = 0;
    explicit Reader(const char* path) {
        f = fopen(path, "r");
        if (!f) throw std::runtime_error(std::string("open: ") + path);
        setvbuf(f, nullptr, _IOFBF, 1 << 20);
    }
    ~Reader() { fclose(f); }
    bool next() {
        while (fgets(line, sizeof(line), f)) {
            ++lines;
            nt = 0;
            char* save = nullptr;
            for (char* s = strtok_r(line, " \t\r\n", &save); s;
                 s = strtok_r(nullptr, " \t\r\n", &save)) {
                if (nt == 8) throw std::runtime_error("too many tokens");
                t[nt++] = s;
            }
            if (nt && t[0][0] != '*') return true;
        }
        return false;
    }
    void get() { if (!next()) throw std::runtime_error("unexpected EOF"); }
    bool is(const char* s) const { return nt && strcmp(t[0], s) == 0; }
};

void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
long long id(const char* text, char prefix) {
    check(text[0] == prefix, std::string("bad name: ") + text);
    char* end;
    long long value = strtoll(text + 1, &end, 10);
    check(*end == '\0', "invalid index");
    return value;
}
double number(const char* text) {
    char* end;
    double v = strtod(text, &end);
    check(*end == '\0' && std::isfinite(v), "invalid finite number");
    return v;
}
long long raw_row(const char* text, int n) {
    if (strncmp(text, "NX", 2) == 0) return id(text + 1, 'X');
    check(text[0] == 'C', "bad raw row");
    char* end;
    long long sub = strtoll(text + 1, &end, 10);
    check(*end == 'A', "bad raw marginal row");
    long long kl = strtoll(end + 1, &end, 10);
    check(*end == '\0', "bad raw marginal suffix");
    return 2 * n + (kl - 1) * 2 * (n - 1) + sub;
}

#include <charconv>

int main(int argc, char** argv) try {
    check(argc == 3, "usage: stream_rescale raw.mps n > scaled.mps");
    int n = atoi(argv[2]);
    long long nr = 2LL*n + 2LL*n*n*(n-1);
    long long nc = 1LL*n*n + 1LL*n*n*(n-1)*(n-1)/2;
    Reader raw(argv[1]);
    setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    auto emit = [](const char* col, const char* row, double value) {
        char text[128];
        auto result = std::to_chars(text, text + sizeof(text) - 1, value);
        check(result.ec == std::errc(), "number formatting");
        *result.ptr = '\0';
        printf("    %-10s%-10s%s\n", col, row, text);
    };
    raw.get(); check(raw.is("NAME"), "NAME");
    raw.get(); check(raw.is("ROWS"), "ROWS");
    raw.get(); check(raw.is("N") && strcmp(raw.t[1], "NOBJ") == 0, "objective row");
    printf("NAME\nROWS\n N  OBJ\n");
    for (long long r = 1; r <= nr; ++r) {
        raw.get();
        check(raw.nt == 2 && raw.is("E") && raw_row(raw.t[1],n) == r, "equality row");
        printf(" L  R%lld\n", r);
    }
    raw.get(); check(raw.is("COLUMNS"), "COLUMNS");
    puts("COLUMNS");
    raw.get();
    std::vector<std::pair<long long,double>> entries;
    for (long long c = 1; c <= nc; ++c) {
        check(raw.nt == 3, "column start");
        std::string name(raw.t[0]);
        entries.clear();
        double cost = 0;
        do {
            check(raw.nt == 3, "column entry");
            if (strcmp(raw.t[1], "NOBJ") == 0) cost = number(raw.t[2]);
            else entries.emplace_back(raw_row(raw.t[1],n), number(raw.t[2]));
            raw.get();
        } while (raw.nt == 3 && name == raw.t[0]);
        std::sort(entries.begin(), entries.end());
        double scale = cost != 0 ? 1.0/std::abs(cost) : 1.0;
        char col[32], row[32];
        snprintf(col, sizeof(col), "C%lld", c);
        for (auto [r,v] : entries) {
            snprintf(row, sizeof(row), "R%lld", r);
            emit(col, row, v*scale);
        }
        if (cost != 0) emit(col, "OBJ", cost*scale);
        if (c % 10000000 == 0) fprintf(stderr, "wrote %lld columns\n", c);
    }
    check(raw.is("RHS"), "RHS");
    for (int r = 1; r <= 2*n; ++r) {
        raw.get();
        check(raw.nt == 3 && raw.is("RHS") && raw_row(raw.t[1],n) == r && number(raw.t[2]) == 1, "assignment RHS");
    }
    raw.get(); check(raw.is("ENDATA") && !raw.next(), "default bounds/end");
    puts("RHS");
    char row[32], col[32];
    for (long long r = 1; r <= nr; ++r) {
        snprintf(row, sizeof(row), "R%lld",r);
        emit("rhs", row, r <= 2*n ? 1 : 0);
    }
    puts("RANGES");
    for (long long r = 1; r <= nr; ++r) {
        snprintf(row, sizeof(row), "R%lld",r);
        emit("rhs", row, 0);
    }
    puts("BOUNDS");
    for (long long c = 1; c <= nc; ++c) {
        snprintf(col, sizeof(col), "C%lld", c);
        printf(" LO %-10s%-10s0\n PL %-10s%s\n", "bounds", col, "bounds", col);
    }
    puts("ENDATA");
    check(fflush(stdout) == 0 && !ferror(stdout), "output write failed");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
}
